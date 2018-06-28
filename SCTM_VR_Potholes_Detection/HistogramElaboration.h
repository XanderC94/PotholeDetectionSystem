#ifndef POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
#define POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H

#include <opencv2/core.hpp>

using namespace cv;

Mat ExtractHistograms(const Mat src, const String candidateName, const int hist_size);

#endif //POTHOLEDETECTIONSYSTEM_HISTOGRAMELABORATION_H
