//
// Created by rui on 19-4-28.
//

#include "Mtcnn.hpp"

vector<float> calculateScales(int width, intt height) {
  vector<float> scales;
  float pr_scale = 1.0;
  if (width > 1000 || height > 1000)
    pr_scale = 1000.0/max(width, height);

  int wd = int(width * pr_scale);
  int hg = int(height*pr_scale);

  float factor = 0.7937;
  int factor_count = 0;
  int minsize = min(wd, hg);
  while (minsize >= 12) {
    scales.push_back(pr_scale*pow(factor, factor_count));
    minsize *= factor;
    factor_count += 1;
  }
  return scales;
}

Mtcnn::Mtcnn(const vector<string> &model_file, const vector<string> &trained_file) {

  #ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  #else
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
  #endif

  CHECK_EQ(model_file.size(), 3) << "size of net_deploy should same with three";
  CHECK_EQ(trained_file.size(), 3) << "size of weight_file should same with three";

  /* Load the network. */
  Pnet_.reset(new Net<float>(model_file[0], TEST));
  Pnet_->CopyTrainedLayersFrom(trained_file[0]);
  Rnet_.reset(new Net<float>(model_file[1], TEST));
  Rnet_->CopyTrainedLayersFrom(trained_file[1]);
  Onet_.reset(new Net<float>(model_file[2], TEST));
  Onet_->CopyTrainedLayersFrom(trained_file[2]);

  /* Set network output blob */
  Pnet_cls_ = Pnet_->blob_by_name("prob1");
  Pnet_roi_ = Pnet_->blob_by_name("conv4-2");
  Rnet_cls_ = Rnet_->blob_by_name("prob1");
  Rnet_roi_ = Rnet_->blob_by_name("conv5-2");
  Onet_cls_ = Onet_->blob_by_name("prob1");
  Onet_roi_ = Onet_->blob_by_name("conv6-2");
  Onet_pts_ = Onet_->blob_by_name("conv6-3");

  input_geometrys_->data()[0] = cv::Size(Pnet_->input_blobs()[0]->width(), Pnet_->input_blobs()[0]->height());
  input_geometrys_->data()[1] = cv::Size(Rnet_->input_blobs()[0]->width(), Rnet_->input_blobs()[0]->height());
  input_geometrys_->data()[2] = cv::Size(Onet_->input_blobs()[0]->width(), Onet_->input_blobs()[0]->height());



}

void Mtcnn::Predict(const cv::Mat& img, NetType netType) {
  boost::shared_ptr<Net<float> > net;
  switch (netType) {
    case NetType::Pnet:
      net_.reset(Pnet_.get());
      input_geometry_ = *(input_geometrys_->data());
      break;
    case NetType::Rnet:
      net_.reset(Rnet_.get());
      input_geometry_ = *(input_geometrys_->data() + 1);
      break;
    case NetType::Onet:
      net_.reset(Onet_.get());
      input_geometry_ = *(input_geometrys_->data() + 2);
      break;
    default:
      std::cout << "not NetTytpe" << std::endl;
      exit(1);
      break;
  }


  Blob<float>* input_layer = net->input_blobs()[0];
  input_layer->Reshape(1, 3,
                       input_geometry_.height, input_geometry_.width);
  net->Reshape();
  vector<cv::Mat> images;
  images.push_back(img);
  Preprocess(images);
}

void Mtcnn::Predict(const vector<cv::Mat> imgs, NetType netType) {
  switch (netType) {
    case NetType::Pnet:
      net_.reset(Pnet_.get());
      input_geometry_ = *(input_geometrys_->data());
      break;
    case NetType::Rnet:
      net_.reset(Rnet_.get());
      input_geometry_ = *(input_geometrys_->data() + 1);
      break;
    case NetType::Onet:
      net_.reset(Onet_.get());
      input_geometry_ = *(input_geometrys_->data() + 2);
      break;
    default:
      std::cout << "not NetTytpe" << std::endl;
      exit(1);
      break;
  }

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, 3,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();
  vector<cv::Mat> images;
  for (int idx = 0; idx < imgs.size(); ++idx)
    images.push_back(imgs[idx]);
  Preprocess(images);
  net_->Forward();
}

void Mtcnn::Preprocess(const vector<cv::Mat>& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  std::vector<cv::Mat> *input_channels;
  int width = input_geometry_.width;
  int height = input_geometry_.height;

  float* input_data = input_layer->mutable_cpu_data();
  for (int img_idx = 0; img_idx < img.size(); ++img_idx) {
    for (int cha_idx = 0; cha_idx < input_layer->channels(); ++cha_idx) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(std::move(channel));
      input_data += width * height;
    }

    /* Convert the input image to the input image format of the network. */
    cv::Mat sample_resized;
    cv::resize(img, sample_resized, input_geometry_);

    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3);

    cv::split(sample_resized, *input_channels);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
  }
}

void Mtcnn::Detection(const cv::Mat &img) {
  int origin_h = img.rows;
  int origin_w = img.cols;
  vector<float> scales = calculateScales(origin_w, origin_h);

  for (auto scale = scales.begin(); scale != scales.end(); ++scale) {
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(int(*scale * origin_w), int(*scale * origin_h)));
    Predict(resize_img, NetType::Pnet);

  }
}

vector<cv::Rect> Mtcnn::Pnet(int origin_w, int origin_h, float scale) {

  const float *cls_data = Pnet_cls_->cpu_data();
  const float *roi_data = Pnet_roi_->cpu_data();
  cls_data = cls_data + Pnet_cls_->offset(0, 1);
  const int out_w = Pnet_cls_->width();
  const int out_h = Pnet_cls_->height();
  const int out_side = max(out_w, out_h);
  int in_side = 2*out_side + 11;
  float stride = 0;
  if (out_side != 1) {
    stride = float(in_side-12) / (out_side - 1);
  }

  for (int h = 0; h < out_h; ++h) {
    for (int w = 0; w < out_w; ++w) {
      float score = *(cls_data++);
      if (score > thresholds_[0]) {
        int original_x1 = int((stride*w+1)*scale);
        int original_y1 = int((stride*h+1)*scale);
        int original_w = int((12.0-1)*scale);
        int original_h = int((12.0-1)*scale);
        int original_x2 = original_x1 + origin_w;
        int original_y2 = original_y1 + origin_h;
        int x1 = int(round(max(0.f,   float(original_x1 + original_w * *(roi_data+Pnet_roi_->offset(0,0,h,w))))));
        int y1 = int(round(max(0.f,   float(original_y1 + original_h * *(roi_data+Pnet_roi_->offset(0,1,h,w))))));
        int x2 = int(round(min(float(origin_w),   float(original_x2 + original_w * *(roi_data+Pnet_roi_->offset(0,2,h,w))))));
        int y2 = int(round(min(float(origin_h),   float(original_y2 + original_h * *(roi_data+Pnet_roi_->offset(0,3,h,w))))));
        if (x2 > x1 && y2 > y1){

        }
      }


    }
  }
}
