//
// Created by Xander_C on 27/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_MLUTILS_H
#define POTHOLEDETECTIONSYSTEM_MLUTILS_H

#include <opencv2/core.hpp>
#include "ontologies.hpp"

using namespace phd::ontologies;

namespace phd::ml::utils {
    Mat ConvertFeatures(const vector<Features> &features);

    Mat ConvertHOGFeatures(const vector<Features> &features, const int var_count);

    Mat ConvertFeaturesForBayes(const vector<Features> &features);

    Mat ConvertFeaturesForSVM(const vector<Features> &features, const int var_count);

    Mat mergeMultiClassifierResults(Mat svmResult, Mat bayesResult);
}

#endif //POTHOLEDETECTIONSYSTEM_MLUTILS_H
