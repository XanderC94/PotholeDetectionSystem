//
// Created by Xander_C on 27/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_BAYES_H
#define POTHOLEDETECTIONSYSTEM_BAYES_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "ontologies.hpp"

namespace phd {
    namespace ml {
        namespace bayes {

                void Classifier(
                        const std::vector<phd::ontologies::Features> &features,
                        cv::Mat &labels,
                        const std::string model_path
                        );

                void Training(
                        const std::vector<phd::ontologies::Features> &features,
                        const cv::Mat &labels,
                        const std::string model_path
                        );

        }
    }
}

#endif //POTHOLEDETECTIONSYSTEM_BAYES_H
