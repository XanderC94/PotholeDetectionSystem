//
// Created by Matteo Gabellini on 02/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H

#include <opencv2/core.hpp>

#include "DataStructures.h"

using namespace cv;

vector<Features> extractFeatures(const Mat &src, const vector<SuperPixel> &candidateSuperPixels,
                                 const Size &candidate_size,
                                 const RoadOffsets &offsets,
                                 const ExtractionThresholds &thresholds);

#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
