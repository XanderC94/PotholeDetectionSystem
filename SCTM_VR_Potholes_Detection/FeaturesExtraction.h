//
// Created by Matteo Gabellini on 02/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_FEATURESEXTRACTION_H
#define POTHOLEDETENCTIONSYSTEM_FEATURESEXTRACTION_H

#include <opencv2/core.hpp>

#include "DataStructures.h"

using namespace cv;

vector<Features> extractFeatures(const Mat &src,
                                 const vector<SuperPixel> &candidateSuperPixels,
                                 const Size candidate_size,
                                 const ExtractionThresholds thresholds);

#endif //POTHOLEDETENCTIONSYSTEM_FEATURESEXTRACTION_H
