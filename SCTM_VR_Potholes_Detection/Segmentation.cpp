#include "Segmentation.h"
#include "HistogramElaboration.h"
#include "MathUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;


int PotholeSegmentation(String img_path,
                        const int SuperPixelEdge,
                        const double Cutline_offset,
                        const double Density_Threshold,
                        const double Variance_Threshold,
                        const double Gauss_RoadThreshold,
                        const double Rects_X_Offset,
                        const double Rects_Y_Offset) {

    Mat src = imread(img_path, IMREAD_COLOR), tmp, imgCIELab, contour, labels;

    const int W = 640;
    const int H = 480;

    /*
     * "Downloads/test_rec6.jpg";
     * */
    //"D:\\Xander_C\\Downloads\\test_rec6.jpg";

    Size scale(W, H);
    Point2d translation_((double) -W / 2.0, (double) -H / 1.0), shrink_(3.0 / (double) W, 5.0 / (double) H);

    Mat imgWithGaussianBlur;
    Mat imgGrayScale;

    resize(src, src, scale);

    //ExtractHistograms(src);
    // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
    GaussianBlur(src, tmp, Size(3, 3), 0.0);

    //Show the result of the blur operation
    tmp.copyTo(imgWithGaussianBlur);
    imshow("GaussianBlur", imgWithGaussianBlur);


    // Switch spazio colore da RGB CieLAB
    cvtColor(tmp, imgCIELab, COLOR_BGR2Lab);
    //imshow("CieLab color space", imgCIELab);

    // Switch spazio di colore da RGB a GreayScale
    cvtColor(tmp, imgGrayScale, CV_BGR2GRAY);
    //imshow("Grey scale", imgGrayScale);
    ExtractHistograms(imgGrayScale);

    // Linear Spectral Clustering
    Ptr<SuperpixelLSC> superpixels = cv::ximgproc::createSuperpixelLSC(imgCIELab, SuperPixelEdge);
    //Ptr<SuperpixelSLIC> superpixelSegmentation = cv::ximgproc::createSuperpixelSLIC(img1, SLIC::MSLIC, 32, 50.0);

    superpixels->iterate(10);
    superpixels->getLabelContourMask(contour);
    superpixels->getLabels(labels);
/*
	src.copyTo(tmp);
	tmp.setTo(Scalar(0, 0, 0), contour);
	imshow("Contours", tmp);
*/
    Mat out, mask, res;
    src.copyTo(out);
    src.copyTo(mask);
    mask.setTo(Scalar(255, 255, 255));

    cout << "SP, Size, Area, Variance, Density" << endl;

    for (int l = 0; l < superpixels->getNumberOfSuperpixels(); ++l) {

        Mat1b LabelMask = (labels == l);

        vector<cv::Point> PixelsInLabel;
        cv::Point Center(0.0, 0.0);
        cv::Point2d Variance(0.0, 0.0);
        double Density = 0.0;

        findNonZero(LabelMask, PixelsInLabel);

        for (auto p : PixelsInLabel) Center += p;
        Center /= (double) PixelsInLabel.size();

        //Point translated_((Center.x + translation_.x), (Center.y + translation_.y));
        //Point shrinked_(translated_.x*shrink_.x, translated_.x*shrink_.y);

        //cout << "Superpixel center @ " << SuperPixel << ", translated @ " << translated_ << ", shrinked @ " << shrinked_ << endl;

        // How to evaluate the thresholds in order to separate road super pixels?
        // Or directly identify RoI where a pothole willl be more likely detected?
        // ...
        // Possibilities
        // => Gaussian 3D function
        //auto gVal = GaussianEllipseFunction3D(shrinked_) > Gauss_RoadThreshold;

        // => Evaluates the pixels through an Analytic Rect Function : F(x) > 0 is over the rect, F(x) < 0 is under, F(x) = 0 it lies on.

        bool isRoad = AnalyticRect2D(Point(W * Rects_X_Offset, H * Rects_Y_Offset),
                                     Point(W * 0.5, H * (Cutline_offset - 0.1)), Center) >= 0 &&
                      AnalyticRect2D(Point(W * (1.0 - Rects_X_Offset), H * Rects_Y_Offset),
                                     Point(W * 0.5, H * (Cutline_offset - 0.1)), Center) >= 0;

        bool isNotOverHorizon = Center.y >= (1 - Cutline_offset) * H;

        Scalar mean_color_value = mean(src, LabelMask);
        Scalar color_mask_value = Scalar(0, 0, 0);

        if (isRoad && isNotOverHorizon) {

            for (Point2d p : PixelsInLabel) {
                Variance = p - (Point2d) Center;
                Variance = Point(Variance.x * Variance.x, Variance.y * Variance.y);
            }

            Variance /= (double) PixelsInLabel.size();
            Point2f verts[4];
            minAreaRect(PixelsInLabel).points(verts);
            // Shoelace Area Formula
            double Area =
                    ((verts[0].x * verts[1].y - verts[1].x * verts[0].y) +
                     (verts[1].x * verts[2].y - verts[2].x * verts[1].y) +
                     (verts[2].x * verts[3].y - verts[3].x * verts[2].y) +
                     (verts[3].x * verts[0].y - verts[0].x * verts[3].y)) * 0.5;
            Density = (double) PixelsInLabel.size() / Area;

            if (Density < Density_Threshold && (Variance.x > Variance_Threshold || Variance.y > Variance_Threshold)) {
                cout << l << ", " << PixelsInLabel.size() << ", " << Area << ", \"" << Variance << "\", " << Density
                     << endl;

                // mean_color_value = mean(src, LabelMask);
                color_mask_value = Scalar(255, 255, 255);
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

    //dilate(mask, mask, dilateElem);
    erode(mask, mask, erodeElem);

    //imshow("Mask", mask);

    src.copyTo(res, mask);

    imshow("Result", res);

    // Cut the image in order to resize it to the smalled square/rectangle possible

    waitKey();

    return 1;
}