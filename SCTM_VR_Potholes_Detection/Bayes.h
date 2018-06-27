//
// Created by Xander_C on 27/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_BAYES_H
#define POTHOLEDETECTIONSYSTEM_BAYES_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

namespace myBayes {

    void Classifier(const Mat &features, Mat &labels, const string model_path);

    void Training(const Mat &features, const Mat &labels, const string model_path);

}

#endif //POTHOLEDETECTIONSYSTEM_BAYES_H
