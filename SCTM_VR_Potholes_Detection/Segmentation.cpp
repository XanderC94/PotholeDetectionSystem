#include "Segmentation.h"
#include "MathUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;


int PotholeSegmentation(Mat &src,
						vector<Point> &candidates,
						const int SuperPixelEdge,
						const double Horizon_Offset,
						const double Density_Threshold,
						const double Variance_Threshold,
						const double Gauss_RoadThreshold,
						const double SLine_X_Offset,
						const double SLine_Y_Offset) {

    Mat tmp, imgCIELab, contour, labels;

    const int W = 640;
    const int H = 480;
	
    Size scale(W, H);
//    Point2d translation_((double) -W / 2.0, (double) -H / 1.0), shrink_(3.0 / (double) W, 5.0 / (double) H);

    resize(src, src, scale);

	src = src(Rect(Point2d(0, H*Horizon_Offset), Point2d(W - 1, H - 1)));

//	imshow("Crop", src);
	
	// Switch color space from RGB to CieLAB
	cvtColor(src, imgCIELab, COLOR_BGR2Lab);
//	imshow("CieLab color space", imgCIELab);

    // Linear Spectral Clustering
    Ptr<SuperpixelLSC> superpixels = cv::ximgproc::createSuperpixelLSC(imgCIELab, SuperPixelEdge);
//    Ptr<SuperpixelSLIC> superpixelSegmentation = cv::ximgproc::createSuperpixelSLIC(imgCIELab, SLIC::MSLIC, 32, 50.0);

    superpixels->iterate(12);
    superpixels->getLabelContourMask(contour);
    superpixels->getLabels(labels);

    Mat out, mask, res;
    src.copyTo(out);
    src.copyTo(mask);
    mask.setTo(Scalar(255, 255, 255));

    cout << "SP, Size, Area, Variance, Density" << endl;

    for (int l = 0; l < superpixels->getNumberOfSuperpixels(); ++l) {

        Mat1b LabelMask = (labels == l);

        vector<cv::Point> PixelsInLabel;
        cv::Point2d Center(0.0, 0.0);
        cv::Point2d Variance(0.0, 0.0);
        double Density = 0.0;

        findNonZero(LabelMask, PixelsInLabel);

        for (Point2d p : PixelsInLabel) Center += p;
        Center /= (double) PixelsInLabel.size();

//        Point translated_((Center.x + translation_.x), (Center.y + translation_.y));
//        Point shrinked_(translated_.x*shrink_.x, translated_.x*shrink_.y);

        // How to evaluate the thresholds in order to separate road super pixels?
        // Or directly identify RoI where a pothole willl be more likely detected?
        // ...
        // Possibilities
        // => Gaussian 3D function
        // => Evaluates the pixels through an Analytic Rect Function : F(x) > 0 is over the rect, F(x) < 0 is under, F(x) = 0 it lies on.

        bool isRoad = AnalyticRect2D(cv::Point2d(src.cols * SLine_X_Offset, src.rows * SLine_Y_Offset),
                                     cv::Point2d(src.cols * 0.4, 0.0), Center) >= 0 &&
                      AnalyticRect2D(cv::Point2d(src.cols * (1.0 - SLine_X_Offset), src.rows * SLine_Y_Offset),
                                     cv::Point2d(src.cols * 0.6, 0.0), Center) >= 0;

        Scalar mean_color_value = mean(src, LabelMask);
        Scalar color_mask_value = Scalar(0, 0, 0);

        if (isRoad) {

            for (Point2d p : PixelsInLabel) {
                Variance = p - (cv::Point2d) Center;
                Variance = Point2d(Variance.x * Variance.x, Variance.y * Variance.y);
            }

            Variance /= (double) PixelsInLabel.size();
            Point2f vertex[4];
            minAreaRect(PixelsInLabel).points(vertex);
            // Shoelace Area Formula
            double Area =
                    ((vertex[0].x * vertex[1].y - vertex[1].x * vertex[0].y) +
                     (vertex[1].x * vertex[2].y - vertex[2].x * vertex[1].y) +
                     (vertex[2].x * vertex[3].y - vertex[3].x * vertex[2].y) +
                     (vertex[3].x * vertex[0].y - vertex[0].x * vertex[3].y)) * 0.5;

            Density = (double) PixelsInLabel.size() / Area;

            if (Density < Density_Threshold && (Variance.x > Variance_Threshold || Variance.y > Variance_Threshold)) {
                
				cout << l << ", " << PixelsInLabel.size() << ", " << Area << ", \"" << Variance << "\", " << Density << endl;

                color_mask_value = Scalar(255, 255, 255);

				// Add the center of the superpixel to the candidates.
				candidates.push_back(Center);
            }
        }

        out.setTo(mean_color_value, LabelMask);
        mask.setTo(color_mask_value, LabelMask);
    }

    out.setTo(Scalar(0, 0, 0), contour);

    imshow("Segmentation", out);

    // Dilate to clean possible small black dots into the image "center"
    auto dilateElem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    // Remove smal white dots outside the image "center"
    auto erodeElem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

//    dilate(mask, mask, dilateElem);
    erode(mask, mask, erodeElem);

//    imshow("Mask", mask);

    src.copyTo(res, mask);

    imshow("Result", res);

    // Cut the image in order to resize it to the smalled square/rectangle possible
    // To Do ...
//    waitKey();

    return 1;
}