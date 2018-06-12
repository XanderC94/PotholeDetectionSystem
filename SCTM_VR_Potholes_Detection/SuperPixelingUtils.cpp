//
// Created by Matteo Gabellini on 11/06/2018.
//

#include "SuperPixelingUtils.h"


SuperPixel getSuperPixel(Mat src,
                         int superPixelLabel,
                         Mat labels) {
    Mat1b selectionMask = (labels == superPixelLabel);
    Mat superPixelSelection;
    src.copyTo(superPixelSelection, selectionMask);
    vector<cv::Point> superPixelPoints;
    findNonZero(selectionMask, superPixelPoints);
    Scalar meanColourValue = mean(src, selectionMask);

    SuperPixel result = {
            .points = superPixelPoints,
            .superPixelSelection = superPixelSelection,
            .selectionMask = selectionMask,
            .meanColourValue = meanColourValue
    };

    return result;
}

Ptr<SuperpixelLSC> initSuperPixelingLSC(Mat &src,
                                        Mat &contour,
                                        Mat &mask,
                                        Mat &labels,
                                        vector<Point> &candidates,
                                        int superPixelEdge) {
    Mat imgCIELab;
    // Switch color space from RGB to CieLAB
    cvtColor(src, imgCIELab, COLOR_BGR2Lab);
//	imshow("CieLab color space", imgCIELab);

    // Linear Spectral Clustering
    Ptr<SuperpixelLSC> superpixels = cv::ximgproc::createSuperpixelLSC(imgCIELab, superPixelEdge);
//  Ptr<SuperpixelSLIC> superpixels = cv::ximgproc::createSuperpixelSLIC(imgCIELab, SLIC::MSLIC, 32, 50.0);

    superpixels->iterate(10);
    superpixels->getLabelContourMask(contour);
    superpixels->getLabels(labels);

    src.copyTo(mask);
    mask.setTo(Scalar(255, 255, 255));

//  cout << "SP, Size, Area, Variance, Density" << endl;
    return superpixels;
}


Ptr<SuperpixelSLIC> initSuperPixelingSLIC(Mat &src, Mat &contour, Mat &labels, Mat &mask) {
    Mat imgCIELab;
    // Switch color space from RGB to CieLAB
    cvtColor(src, imgCIELab, COLOR_BGR2Lab);
//	imshow("CieLab color space", imgCIELab);

    int regionSize = 32;
    float ruler = 20.0;
    Ptr<SuperpixelSLIC> superpixels = cv::ximgproc::createSuperpixelSLIC(imgCIELab, SLIC::MSLIC, regionSize, ruler);

    superpixels->iterate(10);
    superpixels->getLabelContourMask(contour);
    superpixels->getLabels(labels);

    src.copyTo(mask);
    mask.setTo(Scalar(255, 255, 255));

    return superpixels;
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