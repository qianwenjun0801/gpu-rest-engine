#include "classification.h"

#include <iosfwd>
#include <vector>

#define USE_CUDNN 1
#include <caffe/caffe.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "gpu_allocator.h"

using namespace caffe;
using std::string;
using GpuMat = cv::cuda::GpuMat;
using namespace cv;

/* Based on the this repo caffe/classification.cpp and
 * ssd_detect.cpp(https://github.com/limjoe/caffe/blob/ssd_bvlc_inference/examples/ssd/ssd_detect.cpp)
 * example of SSD, but with GPU image preprocessing and a simple memory pool. */
class Classifier
{
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_value,
               const string& label_file,
               GPUAllocator* allocator);

    std::vector<vector<float> > Classify(const Mat& img);

private:
    void SetMean(const string& mean_value);

    std::vector<vector<float> > Predict(const Mat& img);

    void WrapInputLayer(std::vector<GpuMat>* input_channels);

    void Preprocess(const Mat& img,
                    std::vector<GpuMat>* input_channels);

private:
    GPUAllocator* allocator_;
    std::shared_ptr<Net<float>> net_;
    Size input_geometry_;
    int num_channels_;
    GpuMat mean_;
    std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_value,
                       const string& label_file,
                       GPUAllocator* allocator)
    : allocator_(allocator)
{
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_ = std::make_shared<Net<float>>(model_file, TEST);
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_value);

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
}

std::vector<vector<float> > Classifier::Classify(const Mat& img){
    std::vector<vector<float> > output = Predict(img);

    return output;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_value)
{
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          Scalar(values[i]));
      channels.push_back(channel);
    }
    merge(channels, mean_);//merge and sort;
}

std::vector<vector<float> > Classifier::Predict(const Mat& img){
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<GpuMat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Wrap the input layer of the network in separate GpuMat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<GpuMat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_gpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat& host_img,
                            std::vector<GpuMat>* input_channels)
{
    GpuMat img(host_img, allocator_);
    /* Convert the input image to the input image format of the network. */
    GpuMat sample(allocator_);
    if (img.channels() == 3 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cuda::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cuda::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    GpuMat sample_resized(allocator_);
    if (sample.size() != input_geometry_)
        cuda::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    GpuMat sample_float(allocator_);
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    GpuMat sample_normalized(allocator_);
    cuda::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the GpuMat
     * objects in input_channels. */
    cuda::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->gpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

/* By using Go as the HTTP server, we have potentially more CPU threads than
 * available GPUs and more threads can be added on the fly by the Go
 * runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
 * when a CPU thread is ready for inference it will try to retrieve an
 * execution context from a queue of available GPU contexts and then do a
 * cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
 * per GPU. */
class ExecContext
{
public:
    friend ScopedContext<ExecContext>;

    static bool IsCompatible(int device)
    {
        cudaError_t st = cudaSetDevice(device);
        if (st != cudaSuccess)
            return false;

        cuda::DeviceInfo info;
        if (!info.isCompatible())
            return false;

        return true;
    }

    ExecContext(const string& model_file,
                 const string& trained_file,
                 const string& mean_value,
                 const string& label_file,
                 int device)
        : device_(device)
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");

        allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
        caffe_context_.reset(new Caffe);
        Caffe::Set(caffe_context_.get());
        classifier_.reset(new Classifier(model_file, trained_file,
                                         mean_value, label_file,
                                         allocator_.get()));
        Caffe::Set(nullptr);
    }

    Classifier* CaffeClassifier()
    {
        return classifier_.get();
    }

private:
    void Activate()
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");
        allocator_->reset();
        Caffe::Set(caffe_context_.get());
    }

    void Deactivate()
    {
        Caffe::Set(nullptr);
    }

private:
    int device_;
    std::unique_ptr<GPUAllocator> allocator_;
    std::unique_ptr<Caffe> caffe_context_;
    std::unique_ptr<Classifier> classifier_;
};

struct classifier_ctx
{
    ContextPool<ExecContext> pool;
};

/* Currently, 2 execution contexts are created per GPU. In other words, 2
 * inference tasks can execute in parallel on the same GPU. This helps improve
 * GPU utilization since some kernel operations of inference will not fully use
 * the GPU. */
constexpr static int kContextsPerDevice = 2;

classifier_ctx* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_value, char* label_file)
{
    try
    {
        ::google::InitGoogleLogging("inference_server");

        int device_count;
        cudaError_t st = cudaGetDeviceCount(&device_count);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not list CUDA devices");

        ContextPool<ExecContext> pool;
        for (int dev = 0; dev < device_count; ++dev)
        {
            if (!ExecContext::IsCompatible(dev))
            {
                LOG(ERROR) << "Skipping device: " << dev;
                continue;
            }

            for (int i = 0; i < kContextsPerDevice; ++i)
            {
                std::unique_ptr<ExecContext> context(new ExecContext(model_file, trained_file,
                                                                    mean_value, label_file, dev));
                pool.Push(std::move(context));
            }
        }

        if (pool.Size() == 0)
            throw std::invalid_argument("no suitable CUDA device");

        classifier_ctx* ctx = new classifier_ctx{std::move(pool)};
        /* Successful CUDA calls can set errno. */
        errno = 0;
        return ctx;
    }
    catch (const std::invalid_argument& ex)
    {
        LOG(ERROR) << "exception: " << ex.what();
        errno = EINVAL;
        return nullptr;
    }
}

const char* classifier_classify(classifier_ctx* ctx,
                                char* buffer, size_t length)
{
    try
    {
        _InputArray array(buffer, length);

        Mat img = imdecode(array, -1);
        if (img.empty())
            throw std::invalid_argument("could not decode image");

        std::vector<vector<float> > predictions;
        {
            /* In this scope an execution context is acquired for inference and it
             * will be automatically released back to the context pool when
             * exiting this scope. */
            ScopedContext<ExecContext> context(ctx->pool);
            auto classifier = context->CaffeClassifier();
            predictions = classifier->Classify(img);
        }

        /* Print the detection results. */
        /* Print the detection results. */
        std::ostringstream os;
        os << "[";
        for (int i = 0; i < predictions.size(); ++i) {
            const vector<float>& d = predictions[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if (score >= 0.1) {
              os << static_cast<int>(d[1]) << " ";
              os << score << " ";
              os << static_cast<int>(d[3] * img.cols) << " ";
              os << static_cast<int>(d[4] * img.rows) << " ";
              os << static_cast<int>(d[5] * img.cols) << " ";
              os << static_cast<int>(d[6] * img.rows) << std::endl;
            }
        }

        errno = 0;
        std::string str = os.str();
        return strdup(str.c_str());
    }
    catch (const std::invalid_argument&)
    {
        errno = EINVAL;
        return nullptr;
    }
}

void classifier_destroy(classifier_ctx* ctx)
{
    delete ctx;
}
