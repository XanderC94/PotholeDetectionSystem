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