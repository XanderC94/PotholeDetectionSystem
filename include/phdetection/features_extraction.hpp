//
// Created by Matteo Gabellini on 02/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
#define POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H

#include <opencv2/core.hpp>
#include "ontologies.hpp"



namespace phd {
    namespace features {

        std::vector<phd::ontologies::Features> extractFeatures(
                const cv::Mat &src,
                const std::vector<phd::ontologies::SuperPixel> &candidateSuperPixels,
                const cv::Size &candidate_size,
                const phd::ontologies::RoadOffsets &offsets,
                const phd::ontologies::ExtractionThresholds &thresholds
        );
    }
}



#endif //POTHOLEDETECTIONSYSTEM_FEATURESEXTRACTION_H
