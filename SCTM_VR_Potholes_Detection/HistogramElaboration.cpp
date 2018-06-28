#include "HistogramElaboration.h"

#include <opencv2/imgproc.hpp>

using namespace std;


Mat ExtractHistograms(const Mat src, const String candidateName, const int hist_size) {
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
    calcHist(&src, 1, 0, Mat(), gs_hist, 1, &hist_size, &histRange, uniform, accumulate);

    transpose(gs_hist, gs_hist);
    // Draw the histograms for B, G and R
//    int hist_w = 512;
//    int hist_h = 400;
//    int bin_w = cvRound((double) hist_w / histSize);
//    Mat tmp;
//    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
//    normalize(gs_hist, tmp, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
//    for (int i = 1; i < histSize; i++) {
//        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(gs_hist.at<float>(i - 1))),
//             Point(bin_w * (i), hist_h - cvRound(tmp.at<float>(i))),
//             Scalar(255, 255, 255), 2, 8, 0);
//    }

    /// Display
//    namedWindow(candidateName + "Grey Scale Histogram", CV_WINDOW_AUTOSIZE);
//    imshow(candidateName + "Grey Scale Histogram", histImage);

    return gs_hist;
}
