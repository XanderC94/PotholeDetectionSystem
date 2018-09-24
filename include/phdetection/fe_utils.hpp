//
// Created by Xander_C on 30/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H

#include "segmentation.hpp"
#include <opencv2/core.hpp>

void calculateContrastEntropyEnergy(float &outContrast,
                                    float &outEntropy,
                                    float &outEnergy,
                                    const phd::ontologies::SuperPixel &candidateSuperPixel,
                                    const cv::Mat &candidateGrayScale);

cv::Mat1f extractHogParams(
        const cv::Mat &sample,
        const cv::Mat &candidateGrayScale,
        const phd::ontologies::SuperPixel &candidateSuperPixel
    );

cv::Mat1b createExclusionMask(
        const cv::Mat &src,
        const cv::Mat &sample,
        const cv::Point2d &tlc,
        const phd::ontologies::RoadOffsets &offsets,
        const phd::ontologies::ExtractionThresholds &thresholds
    );

#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
