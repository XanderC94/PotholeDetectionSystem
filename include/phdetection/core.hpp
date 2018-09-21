//
// Created by Xander on 19/9/2018.
//

#ifndef POTHOLEDETECTION_POTHOLEDETECTION_H
#define POTHOLEDETECTION_POTHOLEDETECTION_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "segmentation.hpp"
#include "features_extraction.hpp"
#include "svm.hpp"
#include "bayes.hpp"
#include "io.hpp"
#include "ml_utils.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace phd::ml::utils;

namespace phd {

    enum __Method { SVM, BAYES, MULTI };

    const map<__Method, string> METHODS = {
            pair<__Method, string>(SVM, "-svm"),
            pair<__Method, string>(BAYES, "-bayes"),
            pair<__Method, string>(MULTI, "-multi"),
    };

    class UndefinedMethod : exception {
    private:
        string error;
    public:

        UndefinedMethod(const string msg) {
            this->error = msg;
        }

        const char *what() const throw() override {
            return error.data();
        }

    };

    vector<Features> getFeatures(const string &target, const phd::io::Configuration &config);

    Mat classify(const string &method, const string &svm_model,
                 const string &bayes_model, const vector<Features> &features);
}

#endif //POTHOLEDETECTION_POTHOLEDETECTION_H
