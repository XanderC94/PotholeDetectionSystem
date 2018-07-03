//
// Created by Xander_C on 30/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H

#include "Segmentation.h"
#include <opencv2/core.hpp>

using namespace std;
using namespace cv::ximgproc;

typedef struct FeaturesVectors {
    vector<float> averageGreyLevels = vector<float>();
    vector<float> averageRedLevels = vector<float>();
    vector<float> averageGreenLevels = vector<float>();
    vector<float> averageBlueLevels = vector<float>();
    vector<Mat> histograms = vector<Mat>();
    vector<float> contrasts = vector<float>();
    vector<float> entropies = vector<float>();
    vector<float> skewnesses = vector<float>();
    vector<float> energies = vector<float>();
    vector<Mat1f> hogDescriptors = vector<Mat1f>();
} FeaturesVectors;

void calculateContrastEntropyEnergy(float &outContrast,
                                    float &outEntropy,
                                    float &outEnergy,
                                    const SuperPixel &candidateSuperPixel,
                                    const Mat &candidateGrayScale);

Mat1f extractHogParams(const Mat &sample, const Mat &candidateGrayScale, const SuperPixel &candidateSuperPixel);

Mat1b createExclusionMask(const Mat &src, const Mat &sample, const Point2d &tlc,
                          const RoadOffsets &offsets, const ExtractionThresholds &thresholds);

#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTIONUTILS_H
