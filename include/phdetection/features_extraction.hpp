//
// Created by Matteo Gabellini on 02/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H

#include <opencv2/core.hpp>

#include "ontologies.hpp"

using namespace cv;
using namespace phd::ontologies;

namespace phd::features {
    vector<Features> extractFeatures(const Mat &src, const vector<SuperPixel> &candidateSuperPixels,
                                     const Size &candidate_size,
                                     const RoadOffsets &offsets,
                                     const ExtractionThresholds &thresholds);
}



#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
