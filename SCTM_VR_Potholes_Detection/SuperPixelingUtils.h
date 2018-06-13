//
// Created by Matteo Gabellini on 11/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H
#define POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;


typedef struct SuperPixel {
    vector<cv::Point> points;
    Mat superPixelSelection;
    Mat1b selectionMask;
    Mat contour;
    Scalar meanColourValue;
} SuperPixel;

SuperPixel getSuperPixel(Mat src,
                         int superPixelLabel,
                         Mat labels,
                         Mat contour);

Ptr<SuperpixelLSC> initSuperPixelingLSC(Mat &src,
                                        Mat &contour,
                                        Mat &mask,
                                        Mat &labels,
                                        vector<Point> &candidates,
                                        int superPixelEdge);

Ptr<SuperpixelSLIC> initSuperPixelingSLIC(Mat &src, Mat &contour, Mat &labels, Mat &mask);

Point2d calculateSuperPixelCenter(vector<cv::Point> pixelOfTheSuperPixel);

Point2d calculateSuperPixelVariance(vector<cv::Point> superPixel, Point2d center);

double calculateSuperPixelDensity(vector<cv::Point> superPixel);


#endif //POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H
