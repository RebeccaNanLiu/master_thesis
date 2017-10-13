#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sys/stat.h>


#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file);

    std::vector<float> extract(const cv::Mat& img);

private:
    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);

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

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
std::vector<float> Classifier::extract(const cv::Mat& img) {

    std::vector<float> feature = Predict(img);

    return feature;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);

    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    const boost::shared_ptr<Blob<float> > feature_blob = net_->blob_by_name("pool5");
    const float* feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(0);

    std::vector<float> feature;
    for (int i = 0; i <  feature_blob->count(); i ++){
        feature.push_back(feature_blob_data[i]);
    }

    return feature;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void makeDirectory(string dirname){

#if defined __WIN32
    std::wstring stemp = s2ws(dirname);
	LPCWSTR dir_name = stemp.c_str();

	CreateDirectory(dir_name, NULL);
#else
    const int dir_err = mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err)
    {
        printf("Directory already exists...");
    }

#endif

}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto labels.txt input_folder output_folder" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    string label_file   = argv[4];
    char * filelist = argv[5];
    const string SAVE_FOLDER(argv[6]);

    Classifier classifier(model_file, trained_file, mean_file, label_file);

    string fileline;
    std::vector<string> paths;
    std::vector<cv::Mat> images;
    std::ifstream file_list(filelist);

    while (std::getline(file_list, fileline)){
        fileline.erase(std::remove(fileline.begin(), fileline.end(), '\n'), fileline.end());
        fileline.erase(std::remove(fileline.begin(), fileline.end(), '\r'), fileline.end());
        paths.push_back(fileline);
    }
    file_list.close();

    for (int i = 0; i < paths.size(); i ++){
        cv::Mat img = cv::imread(paths[i]);
        CHECK(!img.empty()) << "Unable to decode image " << fileline;
        images.push_back(img);
    }

    int num_images = images.size();
    std::vector<std::vector<float> > features;
    std::vector<float> feature;
    for (int j = 0; j < num_images; j ++){
        feature = classifier.extract(images[j]);
        features.push_back(feature);
    }

    std::cout << "Calculating similarity..." << std::endl;
    int feat_size = features[0].size();
    std::vector<std::vector<float> > sim_mat;
    for (int a = 0; a < num_images; a++){
        std::vector<float> dist;
        float sum_distances = 0.0f;
        for (int b = 0; b < num_images; b++){
            if (a == b)
            {
                dist.push_back(0.0f);
                continue;
            }

            // Get the L2 distance in the feature space
            float distance = 0.0f;
            for (int c = 0; c < feat_size; c++){
                float val1 = features[a][c];
                float val2 = features[b][c];
                //cout << "c = " << c << ",    feature1 = " << val1 << ",    feature2 = " << val2 << endl;
                float diff = val2 - val1;
                float sq = diff*diff;
                distance += sq;
            }
            distance = sqrtf(distance);

            // Store the distances and normalizer
            float gamma = 0.01f;
            float expDistance = exp(- gamma*distance);
            dist.push_back(expDistance);
            sum_distances += expDistance;
        }
        for (int aa = 0; aa < dist.size(); aa++)
        {
            dist[aa] = dist[aa] / sum_distances;
        }
        sim_mat.push_back(dist);

    }
//    for (int i = 0; i < num_images; i ++ ){
//        for (int j = 0; j < num_images; j ++){
//            std::cout << "S_(" << i << ", " << j << ")" <<"       =        "<< sim_mat[i][j] << std::endl;
//        }
//    }

    std::cout << "Calculating rank..." << std::endl;
    std::vector<float> c;
    std::vector<float> rank;

    for (int i = 0; i < num_images; i++){
        c.push_back(1.0f / num_images);
        rank.push_back(0);
    }

    float beta = 0.99f;
    for (int k = 0; k < 20000; k++)
    {

        for (int j = 0; j < num_images; j++){
            float temp = 0.0f;
            for (int i = 0; i < num_images; i++){

                if (i == j) continue;
                temp += rank[i] * sim_mat[i][j];
            }
            rank[j] = beta*temp + (1.0f - beta)*c[j];
        }
    }

    std::priority_queue<std::pair<float, string> > q;
    for (int i = 0; i < num_images;i++){
        q.push(std::pair<float, string>(rank[i], paths[i]));
    }

    std::priority_queue<std::pair<float, string> > temp_q = q;
    for (int i=0; i < num_images; i ++){
        std::cout << "rank_(" <<i<<") = " << rank[i] << std::endl;
        std::cout << "priority_rank_(" << i <<") = " << temp_q.top().first << "    with image name = " << temp_q.top().second << std::endl;
        temp_q.pop();
    }

    std::cout << "Saving results..." << std::endl;
    string label_name = "results";
    string dirname = SAVE_FOLDER + label_name;
    makeDirectory(dirname);

    std::ofstream outFileList((dirname+".txt").c_str(), ios::trunc);

    int count = 0;
    while (!q.empty() )
    {
        string fpath = q.top().second;
        //std::cout << fpath.substr(38,11) << std::endl;
        outFileList << fpath.substr(42,41) << std::endl;
        cv::Mat im = cv::imread(fpath);
        std::string imageNumber = std::to_string(++count);
        imageNumber = std::string(4 - imageNumber.length(), '0') + imageNumber;
        cv::imwrite(dirname + "/" + imageNumber + ".jpg", im);
        q.pop();
    }

    if (outFileList.is_open())
    {
        outFileList.close();
    }

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
