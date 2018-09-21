#include <opencv2/imgproc.hpp>
#include "../include/phdetection/math.hpp"

double AnalyticRect2D(cv::Point from, cv::Point to, cv::Point evaluationPoint) {

    return (static_cast<double>( evaluationPoint.x) - static_cast<double>( from.x)) /
           (static_cast<double>( to.x) - static_cast<double>( from.x)) -
           (static_cast<double>( evaluationPoint.y) - static_cast<double>( from.y)) /
           (static_cast<double>( to.y) - static_cast<double>( from.y));

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


Point2d calculateTopLeftCorner(Point centroid, Size candidate_size) {
    auto tlc_x = centroid.x - candidate_size.width * 0.5;
    auto tlc_y = centroid.y - candidate_size.height * 0.5;

    auto tlc = Point2d(tlc_x < 0 ? 0 : tlc_x, tlc_y < 0 ? 0 : tlc_y);

    return tlc;
}

Point2d calculateBottomRightCorner(Point centroid, Mat sourceImage, Size candidate_size) {
    auto brc_x = centroid.x + candidate_size.width * 0.5;
    auto brc_y = centroid.y + candidate_size.height * 0.5;

    auto brc = Point2d(brc_x > sourceImage.cols - 1 ? sourceImage.cols : brc_x,
                       brc_y > sourceImage.rows - 1 ? sourceImage.rows : brc_y);
    return brc;
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

float calculateSkewnessGrayImageRegion(Mat image, vector<Point> region, float averageGrayVal) {
    float result = 0.0;
    for (Point coordinates : region) {
        result = result + cbrtf(image.at<uchar>(coordinates) - averageGrayVal);
    }
    result = result / region.size();
    return result;
}