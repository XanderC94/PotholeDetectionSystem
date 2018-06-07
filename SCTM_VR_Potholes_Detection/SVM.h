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

Mat Classifier(vector<Features> &features, int max_iter, string model_path);
void Training(vector<Features> &features, vector<int> &labels, int max_iter, string model_path);

#endif //POTHOLEDETENCTIONSYSTEM_SVMCLASSIFIER_H
