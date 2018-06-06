#ifndef POTHOLEDETENCTIONSYSTEM_MATHUTILS_H
#define POTHOLEDETENCTIONSYSTEM_MATHUTILS_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


double AnalyticRect2D(cv::Point from, cv::Point to, cv::Point evaluationPoint);

double GaussianEllipseFunction3D(cv::Point P,
                                 cv::Point O = cv::Point(0.0, 0.0),
                                 double SigmaX = 1.0,
                                 double SigmaY = 1.0,
                                 double A = 1.0,
                                 double Theta = 0.0);

float calculateSkewnessGrayImage(Mat image, float averageColorVal);

#endif //POTHOLEDETENCTIONSYSTEM_MATHUTILS_H
