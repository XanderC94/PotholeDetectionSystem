//
// Created by Matteo Gabellini on 11/06/2018.
//

#ifndef POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H
#define POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <set>
#include "DataStructures.h"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

SuperPixel getSuperPixel(const Mat &src, int superPixelLabel,
                         const Mat &labels, const RoadOffsets offsets);

bool isRoad(const int H, const int W, const RoadOffsets offsets, const Point2d center);

Ptr<SuperpixelLSC> initSuperPixelingLSC(const Mat &src, int superPixelEdge);

Ptr<SuperpixelSLIC> initSuperPixelingSLIC(Mat &src, Mat &contour, Mat &labels, Mat &mask);

Point2d calculateSuperPixelCenter(vector<cv::Point> pixelOfTheSuperPixel);

Point2d calculateSuperPixelVariance(vector<cv::Point> superPixel, Point2d center);

double calculateSuperPixelDensity(vector<cv::Point> superPixel);


#endif //POTHOLEDETENCTIONSYSTEM_SUPERPIXELINGUTILS_H
