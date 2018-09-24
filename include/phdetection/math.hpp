#ifndef POTHOLEDETECTIONSYSTEM_MATHUTILS_H
#define POTHOLEDETECTIONSYSTEM_MATHUTILS_H

#include <opencv2/core.hpp>

namespace phd::math {
    double AnalyticRect2D(cv::Point from, cv::Point to, cv::Point evaluationPoint);

    double GaussianEllipseFunction3D(cv::Point P,
                                     cv::Point O = cv::Point(0.0, 0.0),
                                     double SigmaX = 1.0,
                                     double SigmaY = 1.0,
                                     double A = 1.0,
                                     double Theta = 0.0);

    cv::Point2d calculateTopLeftCorner(cv::Point centroid, cv::Size candidate_size);

    cv::Point2d calculateBottomRightCorner(cv::Point centroid, cv::Mat sourceImage, cv::Size candidate_size);

    float calculateSkewnessGrayImage(cv::Mat image, float averageColorVal);

    float calculateSkewnessGrayImageRegion(cv::Mat image,  std::vector<cv::Point> region, float averageGrayVal);
}


#endif //POTHOLEDETECTIONSYSTEM_MATHUTILS_H
