#ifndef POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
#define POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H

#include <opencv2/core.hpp>

using namespace cv;

namespace phd::features {

    Mat histogram(const Mat &src, const Mat &mask, const int hist_size);

    Mat plotHistogram(const Mat hist_values, const int hist_size, const int hist_w = 512, const int hist_h = 400);

}

#endif //POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
