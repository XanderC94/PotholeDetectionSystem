//
// Created by Matteo Gabellini on 15/06/2018.
//

#include "GradientElaboration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

/*
 *
 */
Mat calculateGradientModule(Mat gradientX, Mat gradientY) {
    Mat module;
    gradientX.copyTo(module);

    for (int x = 0; x < gradientX.cols; x++) {
        for (int y = 0; y < gradientX.rows; y++) {
            module.at<uchar>(y, x) = (uchar) sqrt(
                    (pow(gradientX.at<uchar>(y, x), 2) + pow(gradientY.at<uchar>(y, x), 2)));
        }
    }

    return module;
}

Gradient calculateGradient(Mat &candidate) {
    Mat resultx;
    Mat resulty;
    //cv::Sobel(candidate, result, CV_64F, 0 , 1, 5);
    cv::spatialGradient(candidate, resultx, resulty);
    Mat module = calculateGradientModule(resultx, resulty);
    Gradient result = {resultx, resulty, module};
    return result;
}

