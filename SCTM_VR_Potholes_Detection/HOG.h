//
// Created by Xander_C on 15/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_HOG_H
#define POTHOLEDETENCTIONSYSTEM_HOG_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

typedef struct HOGConfig {
    const int bin;
    const Size window;
    const Size block;
    const Size block_stride;
    const Size cell;
    const Size window_stride;
    const Size padding;
} HOGConfig;

const HOGConfig defaultConfig = {9, Size(64, 64), Size(8, 8), Size(2, 2), Size(4, 4), Size(2, 2), Size(0, 0)};

void HOG (const Mat &src, vector<float> &descriptors, vector<Point> &locations, const HOGConfig config = defaultConfig);

Mat get_hogdescriptor_visual_image (const Mat& src, const vector<float>& descriptors,
                                   const Size window, const Size cell, const int bins,
                                   const int scaling_factor = 1, const double grad_viz = 5,
                                   const bool isGrayscale = false, bool onlyMaxGrad = true);

#endif //POTHOLEDETENCTIONSYSTEM_HOG_H