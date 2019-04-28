//
// Created by rui on 19-4-28.
//

#ifndef MAIN_MTCNN_HPP
#define MAIN_MTCNN_HPP

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

using namespace caffe;
using namespace std;

enum NetType{Pnet, Rnet, Onet};

class Mtcnn {
 public:
  Mtcnn(const vector<string>& model_file,
             const vector<string>& trained_file);

  void Classify(const cv::Mat& img);
  void Detection(const cv::Mat &img);

 private:
  void SetMean(const string& mean_file);

  void Predict(const cv::Mat& img, NetType netType);

  void Predict(const vector<cv::Mat> imgs, NetType netType);

  vector<cv::Rect> Pnet(int origin_w, int origin_h, float scale);

  void Preprocess(const vector<cv::Mat>& img);


 private:
  boost::shared_ptr<Net<float> > Pnet_;
  boost::shared_ptr<Net<float> > Rnet_;
  boost::shared_ptr<Net<float> > Onet_;
  vector<float> thresholds_;
  vector<cv::Size> input_geometrys_[3];
  int num_channels_;
  boost::shared_ptr<Blob<float> > Pnet_cls_;
  boost::shared_ptr<Blob<float> > Pnet_roi_;
  boost::shared_ptr<Blob<float> > Rnet_cls_;
  boost::shared_ptr<Blob<float> > Rnet_roi_;
  boost::shared_ptr<Blob<float> > Onet_cls_;
  boost::shared_ptr<Blob<float> > Onet_roi_;
  boost::shared_ptr<Blob<float> > Onet_pts_;

  boost::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;


};


#endif //MAIN_MTCNN_HPP
