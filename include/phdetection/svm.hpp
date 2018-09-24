//
// Created by Xander_C on 5/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
#define POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "ontologies.hpp"

namespace phd::ml::svm {

    void Classifier(const std::vector<phd::ontologies::Features> &features, cv::Mat &labels, const std::string model_path);

    void Training(const std::vector<phd::ontologies::Features> &features,
                  const cv::Mat &labels, const int max_iter, const double epsilon,
                  const std::string model_path
                );
}

#endif //POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
