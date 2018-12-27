//
// Created by Xander_C on 27/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_MLUTILS_H
#define POTHOLEDETECTIONSYSTEM_MLUTILS_H

#include <opencv2/core.hpp>
#include "ontologies.hpp"

namespace phd {
    namespace ml {
        namespace utils {
            cv::Mat ConvertFeatures(const std::vector<phd::ontologies::Features> &features);

            cv::Mat ConvertHOGFeatures(const std::vector<phd::ontologies::Features> &features, const int var_count);

            cv::Mat ConvertFeaturesForBayes(const std::vector<phd::ontologies::Features> &features);

            cv::Mat ConvertFeaturesForSVM(const std::vector<phd::ontologies::Features> &features, const int var_count);

            cv::Mat mergeMultiClassifierResults(cv::Mat svmResult, cv::Mat bayesResult);
        }
    }
}

#endif //POTHOLEDETECTIONSYSTEM_MLUTILS_H
