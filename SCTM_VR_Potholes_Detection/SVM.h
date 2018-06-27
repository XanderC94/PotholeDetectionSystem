//
// Created by Xander_C on 5/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
#define POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

namespace mysvm {
    void Classifier(const Mat &features, Mat &labels, const int max_iter, const string model_path);

    void Training(const Mat &features, const Mat &labels, const int max_iter, const string model_path);
}

#endif //POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
