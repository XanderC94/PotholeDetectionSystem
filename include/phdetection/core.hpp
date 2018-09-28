//
// Created by Xander on 19/9/2018.
//

#ifndef POTHOLEDETECTION_POTHOLEDETECTION_H
#define POTHOLEDETECTION_POTHOLEDETECTION_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "io.hpp"

namespace phd {

    enum __Method { SVM = 0, BAYES = 1, MULTI = 2};

    const std::map<int, std::string> METHODS = {
            std::pair<int, std::string>(SVM, "-svm"),
            std::pair<int, std::string>(BAYES, "-bayes"),
            std::pair<int, std::string>(MULTI, "-multi"),
    };

    class UndefinedMethod : std::exception {
    private:
        std::string error;
    public:

        UndefinedMethod(const std::string msg) {
            this->error = msg;
        }

        const char *what() const throw() override {
            return error.data();
        }

    };

    std::vector<phd::ontologies::Features> getFeatures(const std::string &target, const phd::io::Configuration &config);

    std::vector<phd::ontologies::Features> getFeatures(cv::Mat &target, const phd::io::Configuration &config);

    cv::Mat classify(const std::string &method, const std::string &svm_model,
                 const std::string &bayes_model, const std::vector<phd::ontologies::Features> &features);
}

#endif //POTHOLEDETECTION_POTHOLEDETECTION_H
