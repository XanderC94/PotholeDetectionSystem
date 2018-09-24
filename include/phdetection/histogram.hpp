#ifndef POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
#define POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H

#include <opencv2/core.hpp>

namespace phd::features {

    cv::Mat histogram(const cv::Mat &src, const cv::Mat &mask, const int hist_size);

    cv::Mat plotHistogram(const cv::Mat hist_values, const int hist_size, const int hist_w = 512, const int hist_h = 400);

}

#endif //POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
