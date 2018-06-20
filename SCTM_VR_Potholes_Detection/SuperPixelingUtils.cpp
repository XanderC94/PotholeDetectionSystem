//
// Created by Matteo Gabellini on 11/06/2018.
//

#include "SuperPixelingUtils.h"
#include "MathUtils.h"

bool isRoad(const int H, const int W, const RoadOffsets offsets, const Point2d center) {
    // How to evaluate offsets in order to separate road super pixels?
    // Or directly identify RoI (Region of interest) where a pothole will be more likely detected?
    // ...
    // Possibilities
    // => Gaussian 3D function
    // => Evaluates the pixels through an Analytic Rect Function : F(x) > 0 is over the rect, F(x) < 0 is under, F(x) = 0 it lies on.

    bool isRoad = AnalyticRect2D(cv::Point2d(W * offsets.SLine_X_Left_Offset, H * offsets.SLine_Y_Offset),
                                 cv::Point2d(W * 0.4, 0.0), center) >= -0.01 &&
                  AnalyticRect2D(
                          cv::Point2d(W * (1.0 - offsets.SLine_X_Right_Offset), H * offsets.SLine_Y_Offset),
                          cv::Point2d(W * 0.6, 0.0), center) >= -0.01;

    return isRoad;
}

Mat getContours(const Mat &mask) {
    vector<vector<Point>> tmp;

    Mat maskContours = Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    findContours(mask, tmp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    drawContours(maskContours, tmp, -1, Scalar(255));

    return maskContours;
}

SuperPixel stub(const Mat &src, const int superPixelLabel, const Mat &mask) {

    vector<cv::Point> superPixelPoints;
    vector<vector<Point>> tmp;

    Mat maskContours = getContours(mask);

    Mat superPixelSelection;
    src.copyTo(superPixelSelection, mask);
    findNonZero(mask, superPixelPoints);
    Scalar meanColourValue = mean(src, mask);

    SuperPixel result = {
            .label = superPixelLabel,
            .points = superPixelPoints,
            .center = calculateSuperPixelCenter(superPixelPoints),
            .selection = superPixelSelection,
            .mask = mask,
            .contour = maskContours,
            .meanColourValue = meanColourValue,
            .neighbors= std::set<int>()
    };

    return result;
}

SuperPixel getSuperPixel(const Mat &src, const int superPixelLabel, const Mat &labels) {

    Mat1b selectionMask = (labels == superPixelLabel);

    return stub(src, superPixelLabel, selectionMask);
}

SuperPixel getSuperPixel(const Mat &src, const int superPixelLabel,
                         const Mat &labels, const RoadOffsets offsets) {

    Mat1b selectionMask = (labels == superPixelLabel);

    vector<cv::Point> superPixelPoints;
    findNonZero(selectionMask, superPixelPoints);

    // delete all the mask pixels that are outside the boundaries
    for (auto p : superPixelPoints) {
        if (!isRoad(src.rows, src.cols, offsets, p)) {
            selectionMask.at<uchar>(p) = 0;
        }
    }

    return stub(src, superPixelLabel, selectionMask);
}

Ptr<SuperpixelLSC> initSuperPixelingLSC(const Mat &src, const int superPixelEdge) {
    Mat imgCIELab;

    // Switch color space from RGB to CieLAB
    cvtColor(src, imgCIELab, COLOR_BGR2Lab);

    // Linear Spectral Clustering
    return cv::ximgproc::createSuperpixelLSC(imgCIELab, superPixelEdge);
}


Ptr<SuperpixelSLIC> initSuperPixelingSLIC(const Mat &src, const int superPixelEdge, float ruler) {
    Mat imgCIELab;
    // Switch color space from RGB to CieLAB
    cvtColor(src, imgCIELab, COLOR_BGR2Lab);

    return cv::ximgproc::createSuperpixelSLIC(imgCIELab, SLIC::MSLIC, superPixelEdge, ruler);

}


Point2d calculateSuperPixelCenter(vector<cv::Point> pixelOfTheSuperPixel) {
    cv::Point2d center(0.0, 0.0);
    for (Point2d p : pixelOfTheSuperPixel) center += p;
    center /= (double) pixelOfTheSuperPixel.size();
    return center;
}

Point2d calculateSuperPixelVariance(vector<cv::Point> superPixel, Point2d center) {

    cv::Point2d variance(0.0, 0.0);

    for (Point2d p : superPixel) {
        variance = p - (cv::Point2d) center;
        variance = Point2d(variance.x * variance.x, variance.y * variance.y);
    }

    variance /= (double) superPixel.size();

    return variance;
}

double calculateSuperPixelDensity(vector<cv::Point> superPixel) {
    Point2f vertex[4];
    minAreaRect(superPixel).points(vertex);
    // Shoelace Area Formula
    double area =
            ((vertex[0].x * vertex[1].y - vertex[1].x * vertex[0].y) +
             (vertex[1].x * vertex[2].y - vertex[2].x * vertex[1].y) +
             (vertex[2].x * vertex[3].y - vertex[3].x * vertex[2].y) +
             (vertex[3].x * vertex[0].y - vertex[0].x * vertex[3].y)) * 0.5;


    return (double) superPixel.size() / area;
}