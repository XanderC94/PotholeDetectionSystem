//
// Created by Matteo Gabellini on 11/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H
#define POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <set>
#include "DataStructures.h"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

Mat getContours(const Mat &mask);

SuperPixel getSuperPixel(const Mat &src, int superPixelLabel,
                         const Mat &labels);

SuperPixel getSuperPixel(const Mat &src, int superPixelLabel,
                         const Mat &labels, const RoadOffsets offsets);

SuperPixel getSuperPixel(const Mat &src, const Mat1b &roadMask,
                         const int superPixelLabel, const Mat &labels);

bool isRoad(const int H, const int W, const RoadOffsets offsets, const Point2d center);

Ptr<SuperpixelLSC> initSuperPixelingLSC(const Mat &src, const int superPixelEdge);

Ptr<SuperpixelSLIC> initSuperPixelingSLIC(const Mat &src, const int superPixelEdge, const float ruler);

Point2d calculateSuperPixelCenter(vector<cv::Point> pixelOfTheSuperPixel);

Point2d calculateSuperPixelVariance(vector<cv::Point> superPixel, Point2d center);

double calculateSuperPixelDensity(vector<cv::Point> superPixel);


#endif //POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H
