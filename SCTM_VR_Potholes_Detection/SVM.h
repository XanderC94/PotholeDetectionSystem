//
// Created by Xander_C on 5/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
#define POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

#include "FeaturesExtraction.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

Mat ConvertFeatures(const vector<Features> &features);
void Classifier(const vector<Features> &features, const int max_iter, const string model_path, Mat &labels);
void Training(const vector<Features> &features, const Mat &labels, const int max_iter, const string model_path);

#endif //POTHOLEDETECTIONSYSTEM_SVMCLASSIFIER_H
