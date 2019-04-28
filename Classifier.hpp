//
// Created by rui on 19-3-31.
//

#ifndef MAIN_CLASSIFIER_HPP
#define MAIN_CLASSIFIER_HPP
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  void SetOutputBlob(std::vector<string> blob_name);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 2);

 private:
  void SetMean(const string& mean_file);

  void Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);


 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  boost::shared_ptr<Blob<float> > cls_prob_;
  boost::shared_ptr<Blob<float> > roi_;
};


#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}


#endif //MAIN_CLASSIFIER_HPP
