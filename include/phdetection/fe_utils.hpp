//
// Created by Xander_C on 30/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H

#include "segmentation.hpp"
#include <opencv2/core.hpp>

using namespace std;
using namespace cv::ximgproc;

void calculateContrastEntropyEnergy(float &outContrast,
                                    float &outEntropy,
                                    float &outEnergy,
                                    const SuperPixel &candidateSuperPixel,
                                    const Mat &candidateGrayScale);

Mat1f extractHogParams(const Mat &sample, const Mat &candidateGrayScale, const SuperPixel &candidateSuperPixel);

Mat1b createExclusionMask(const Mat &src, const Mat &sample, const Point2d &tlc,
                          const RoadOffsets &offsets, const ExtractionThresholds &thresholds);

#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
