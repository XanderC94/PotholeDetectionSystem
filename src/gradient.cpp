//
// Created by Matteo Gabellini on 15/06/2018.
//

#include "phdetection/gradient.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace phd {
    namespace features {

        void printDirection(Gradient gradient) {
            for (int x = 0; x < gradient.direction.cols; x++) {
                for (int y = 0; y < gradient.direction.rows; y++) {
                    cout << static_cast<int>( gradient.direction.at<uchar>(y, x)) << " ";
                }
                cout << endl;
            }
        }

        Gradient calculateGradient(Mat candidate) {
            Mat imageGrayScale;

            cvtColor(candidate, imageGrayScale, CV_BGR2GRAY);
            imageGrayScale.convertTo(imageGrayScale, CV_32F, 1 / 255.0);

            //calculate horizontal and vertical gradient
            Mat gradientX;
            Mat gradientY;
            cv::Sobel(imageGrayScale, gradientX, CV_32F, 1, 0, 3);
            cv::Sobel(imageGrayScale, gradientY, CV_32F, 0, 1, 3);
            //cv::spatialGradient(candidate, gradientX, gradientY);

            //calculate magnitude and direction
            Mat mag;
            Mat angle;
            cartToPolar(gradientX, gradientY, mag, angle, 1);

            Gradient result = {gradientX, gradientY, mag, angle};
            return result;
        }

    }
}