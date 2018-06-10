#include <opencv2/imgproc.hpp>
#include "MathUtils.h"

double AnalyticRect2D(cv::Point from, cv::Point to, cv::Point evaluationPoint) {

    return ((double) evaluationPoint.x - (double) from.x) / ((double) to.x - (double) from.x) -
           ((double) evaluationPoint.y - (double) from.y) / ((double) to.y - (double) from.y);

}

double GaussianEllipseFunction3D(cv::Point P,
                                 cv::Point O,
                                 double SigmaX,
                                 double SigmaY,
                                 double A,
                                 double Theta) {

    double a = cos(Theta) * cos(Theta) / (2 * SigmaX * SigmaX) + sin(Theta) * sin(Theta) / (2 * SigmaY * SigmaY);
    double b = -sin(2 * Theta) / (2 * SigmaX * SigmaX) + sin(2 * Theta) / (2 * SigmaY * SigmaY);
    double c = sin(Theta) * sin(Theta) / (2 * SigmaX * SigmaX) + cos(Theta) * cos(Theta) / (2 * SigmaY * SigmaY);

    double x = a * (P.x - O.x) * (P.x - O.x) + 2 * b * (P.x - O.x) * (P.y - O.y) + c * (P.y - O.y) * (P.y - O.y);

    return A * exp(-x);

}

float calculateSkewnessGrayImage(Mat image, float averageGrayVal) {
    float result = 0.0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            result = result + cbrtf(image.at<uchar>(i, j) - averageGrayVal);
        }
    }
    result = result / (image.rows * image.cols);
    return result;
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