#include "HistogramElaboration.h"

#include <opencv2/imgproc.hpp>

using namespace std;


Mat ExtractHistograms(const Mat &src, const Mat &mask, const int hist_size) {
    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    //split( src, bgr_planes );

    /// Establish the number of bins
//    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat gs_hist;

    /// Compute the histograms:
    calcHist(&src, 1, 0, mask, gs_hist, 1, &hist_size, &histRange, uniform, accumulate);

    transpose(gs_hist, gs_hist);

    return gs_hist;
}

Mat getHistogramImage(const Mat hist_values, const int hist_size,
                      const int hist_w, const int hist_h) {
    // Draw the histograms for B, G and R

    int bin_w = cvRound(static_cast<double>( hist_w) / hist_size);
    Mat tmp;
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Normalize the result to [ 0, histImage.rows ]
    normalize(hist_values, tmp, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Draw for each channel
    for (int i = 1; i < hist_size; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_values.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(tmp.at<float>(i))),
             Scalar(255, 255, 255), 2, 8, 0);
    }

    // Display
    return histImage;
}
