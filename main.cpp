#include "Mtcnn.hpp"

int main() {
  vector<string> models;
  models.push_back("./models/12net.prototxt");
  models.push_back("./models/24net.prototxt");
  models.push_back("./models/48net.prototxt");

  vector<string> weight;
  Mtcnn mtcnn = new Mtcnn()

}