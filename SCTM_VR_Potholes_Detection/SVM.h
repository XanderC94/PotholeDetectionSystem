//
// Created by Xander_C on 5/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_SVMCLASSIFIER_H
#define POTHOLEDETENCTIONSYSTEM_SVMCLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

#include "FeaturesExtraction.h"

using namespace cv;
using namespace std;
using namespace ml;

void Classifier(const vector<Features> &features, const int max_iter, const string model_path, Mat &labels);
void Training(const vector<Features> &features, const Mat &labels, const int max_iter, const string model_path);

#endif //POTHOLEDETENCTIONSYSTEM_SVMCLASSIFIER_H
