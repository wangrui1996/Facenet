#include "Mtcnn.hpp"

int main() {
  vector<string> models;
  models.emplace_back("./models/12net.prototxt");
  models.emplace_back("./models/24net.prototxt");
  models.emplace_back("./models/48net.prototxt");

  vector<string> weights;
  weights.emplace_back("./models/12net.caffemodel");
  weights.emplace_back("./models/24net.caffemodel");
  weights.emplace_back("./models/48net.caffemodel");
  Mtcnn *mtcnn = new Mtcnn(models, weights);
  cv::Mat img = cv::imread("./face.jpg");
  mtcnn->Detection(img);

}