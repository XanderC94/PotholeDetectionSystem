//
// Created by Matteo Gabellini on 15/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_GRADIENTELABORATION_H
#define POTHOLEDETENCTIONSYSTEM_GRADIENTELABORATION_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

typedef struct Gradient {
    Mat x;
    Mat y;
    Mat module;
} Gradient;

Gradient calculateGradient(Mat &candidate);

#endif //POTHOLEDETENCTIONSYSTEM_GRADIENTELABORATION_H
