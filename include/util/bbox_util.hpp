#ifndef MAIN_BBOX_UTIL_HPP
#define MAIN_BBOX_UTIL_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

typedef vector<pair<Rect, float>> BoundingBox;

void ApplyNMSFast(vector<pair<Rect, float> > &boundingbox, float threshold);


#endif //MAIN_BBOX_UTIL_HPP
