
#include "util/bbox_util.hpp"

bool SortScorePairDescend(const pair<Rect, float>& pair1,
                          const pair<Rect, float>& pair2) {
    return pair1.second > pair2.second;
}

void GetMaxScoreIndex(vector<pair<Rect, float> > &boundingbox) {
    std::stable_sort(boundingbox.begin(), boundingbox.end(),SortScorePairDescend);
}

float BBoxSize(const cv::Rect bbox) {
    if (bbox.height <= 0 || bbox.width <= 0)
        return 0.f;
    return bbox.area();
}

float JaccardOverlap(const Rect bbox1, const Rect bbox2) {
    if (bbox1.x > (bbox2.x + bbox2.width) || bbox1.x + bbox1.width < bbox2.x ||
    bbox1.y > (bbox2.y + bbox2.height) || bbox1.y + bbox1.height < bbox2.y)
        return 0.f;
    int inter_xmin = std::max(bbox1.x, bbox2.x);
    int inter_ymin = std::max(bbox1.y, bbox2.y);
    int inter_xmax = std::max(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    int inter_ymax = std::max(bbox1.y + bbox1.height, bbox2.y + bbox2.height);
    int inter_size = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin);
    return inter_size / (bbox1.area() + bbox2.area() - inter_size);
}

void ApplyNMSFast(vector<pair<Rect, float> > &boundingbox, float threshold) {
    GetMaxScoreIndex(boundingbox);
    auto bbox_idx = boundingbox.begin();
    while (bbox_idx != boundingbox.end()) {
        auto compare_idx = bbox_idx + 1;
        while (compare_idx != boundingbox.end()) {
            if (JaccardOverlap(bbox_idx->first, compare_idx->first) < threshold)
                compare_idx = boundingbox.erase(compare_idx);
            else
                compare_idx = compare_idx + 1;
        }
    }


}
