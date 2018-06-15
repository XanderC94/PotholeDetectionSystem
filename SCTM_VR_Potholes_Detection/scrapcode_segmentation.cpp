//
// Created by Xander_C on 15/06/2018.
//
#include <opencv2/core.hpp>
#include "DataStructures.h"
#include "SuperPixelingUtils.h"
#include "MathUtils.h"
#include "Segmentation.h"

#include <iostream>

using namespace cv;
using namespace std;

void RoadSegmentation(const Mat &src, Mat &out) {

    src.copyTo(out);

    preprocessing(out, out, 0.2);

    Scalar mean, std;

    meanStdDev(out, mean, std);

    cout << "STD: " << std << endl;
    cout << "MEAN: " << mean << endl;

    for (int row = 0; row < out.rows; ++row) {
        for (int col = 0; col < out.cols; ++col) {

            Vec3b & pixel = out.at<Vec3b>(row, col);

            if (pixel.val[1] - (pixel.val[0] + pixel.val[2]) / 2.0f + 15.0f > 0.0f ||
                pixel.val[0] - pixel.val[2] > 20.0f ||
                pixel.val[1] - pixel.val[0] > 15.0f) {

                out.at<Vec3b>(row, col) =  Vec3b(0,0,0);

            }

            else if ((pixel[0] > 255 - std.val[0] && pixel[1] > 255 - std.val[1] && pixel[2] > 255 - std.val[2]) ||
                     (pixel[0] < std.val[0] && pixel[1] < std.val[0] && pixel[2] < std.val[0])
                    ) {

                out.at<Vec3b>(row, col) = Vec3b(0,0,0);

            } else if (pixel[0] > 0 && pixel[1] > 0 && pixel[2] > 0) {

                out.at<Vec3b>(row, col) = Vec3b(255,255,255);

            }
        }
    }
}