#include "Segmentation.h"
#include "MathUtils.h"

#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

const int RESIZING_WIDTH = 640;
const int RESIZING_HEIGHT = 480;


void preprocessing(Mat &src, Mat &resizedImage, const double Horizon_Offset) {

    cout << "Preprocessing... ";

    Size scale(RESIZING_WIDTH, RESIZING_HEIGHT);
//    Point2d translation_((double) -W / 2.0, (double) -H / 1.0), shrink_(3.0 / (double) W, 5.0 / (double) H);

    // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
    GaussianBlur(src, src, Size(7, 7), 0.0);

    resize(src, resizedImage, scale);

    resizedImage = resizedImage(
            Rect(Point2d(0, RESIZING_HEIGHT * Horizon_Offset), Point2d(RESIZING_WIDTH - 1, RESIZING_HEIGHT - 1)));

    cout << "Finished." << endl;
}

bool isRoad(Mat src, RoadOffsets offsets, Point2d center) {
    // How to evaluate offsets in order to separate road super pixels?
    // Or directly identify RoI (Region of interest) where a pothole will be more likely detected?
    // ...
    // Possibilities
    // => Gaussian 3D function
    // => Evaluates the pixels through an Analytic Rect Function : F(x) > 0 is over the rect, F(x) < 0 is under, F(x) = 0 it lies on.

    bool isRoad = AnalyticRect2D(cv::Point2d(src.cols * offsets.SLine_X_Offset, src.rows * offsets.SLine_Y_Offset),
                                 cv::Point2d(src.cols * 0.4, 0.0), center) >= 0 &&
                  AnalyticRect2D(
                          cv::Point2d(src.cols * (1.0 - offsets.SLine_X_Offset), src.rows * offsets.SLine_Y_Offset),
                          cv::Point2d(src.cols * 0.6, 0.0), center) >= 0;

    return isRoad;
}

/*
 * Check if the super pixel is a pothole:
 *      - checking if its density il less than the specified density threshold
 *      - checking if the variance on x axis or y axis is greater than the variance threshold
 * */
bool isPothole(SuperPixel superPixel, cv::Point2d center, ExtractionThresholds thresholds) {
    cv::Point2d Variance(0.0, 0.0);
    Variance = calculateSuperPixelVariance(superPixel.points, center);
    double density = calculateSuperPixelDensity(superPixel.points);
    return (density < thresholds.Density_Threshold &&
            (Variance.x > thresholds.Variance_Threshold || Variance.y > thresholds.Variance_Threshold));
}


/*
 * Check if:
 *  1) The super pixel is a pothole checking its mean colour value is:
 *      - less than the mean colour value of the candidate and  greater than mean colour value of the candidate / 1.5;
 *      - less than the previous selected super pixel mean value;
 *  2) The number of point of the super pixel is greater than 16*16
 * */
bool isPothole(SuperPixel superPixel, SuperPixel previousSelected, Scalar meanCandidateColourValue) {
    double meanDivider = 1.5;
    double minPixelNumber = 256;
    return superPixel.meanColourValue[0] < meanCandidateColourValue[0]
           && superPixel.meanColourValue[0] > (meanCandidateColourValue[0] / meanDivider)
           && superPixel.meanColourValue[0] < (previousSelected.meanColourValue[0])
           && superPixel.points.size() > minPixelNumber;
}

void extractCandidateCentroids(Mat &src,
                               Mat labels,
                               int nSuperPixels,
                               Mat contour,
                               vector<Point> &candidateCentroids,
                               ExtractionThresholds thresolds,
                               Mat &meanColourMask,
                               Mat &candidateMask,
                               RoadOffsets offsets) {

    src.copyTo(meanColourMask);
    for (int l = 0; l < nSuperPixels; ++l) {
//      Point translated_((Center.x + translation_.x), (Center.y + translation_.y));
//      Point shrinked_(translated_.x*shrink_.x, translated_.x*shrink_.y);
        SuperPixel currentSuperPixel = getSuperPixel(src, l, labels, contour);

        cv::Point2d center = calculateSuperPixelCenter(currentSuperPixel.points);
        Scalar color_mask_value = Scalar(0, 0, 0);
        if (isRoad(src, offsets, center)) {
            if (isPothole(currentSuperPixel, center, thresolds)) {
                color_mask_value = Scalar(255, 255, 255);
                // Add the center of the superpixel to the candidateCentroids.
                candidateCentroids.push_back(center);
            }
        }

        meanColourMask.setTo(currentSuperPixel.meanColourValue, currentSuperPixel.selectionMask);
        candidateMask.setTo(color_mask_value, currentSuperPixel.selectionMask);
    }
}

int extractPossiblePotholes(Mat &src,
                            vector<Point> &candidateCentroids,
                            const int superPixelEdge,
                            const ExtractionThresholds thresholds,
                            const RoadOffsets offsets,
                            const string showingWindowPrefix) {
    Mat contour;
    Mat mask;
    Mat labels;
    Ptr<SuperpixelLSC> superpixels = initSuperPixelingLSC(src, contour, mask, labels, candidateCentroids,
                                                          superPixelEdge);

    //imshow("labels", labels);
    Mat out;
    extractCandidateCentroids(src, labels, superpixels->getNumberOfSuperpixels(), contour, candidateCentroids,
                              thresholds, out,
                              mask, offsets);

    //imshow(showingWindowPrefix + " src", src);
    //imshow(showingWindowPrefix + " Mean colour mask", out);
    //imshow(showingWindowPrefix + " contour", contour);
    //imshow(showingWindowPrefix + " mask", mask);

//    out.setTo(Scalar(0, 0, 255), contour);

    //imshow("Segmentation", out);

    // Dilate to clean possible small black dots into the image "center"
//    auto dilateElem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    // Remove small white dots outside the image "center"
//    auto erodeElem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

//    dilate(mask, mask, dilateElem);
//    erode(mask, mask, erodeElem);

    //imshow("Mask", mask);

//    Mat res;
//    src.copyTo(res, mask);
//    src.copyTo(res, out);

//    imshow(showingWindowPrefix + " Result", res);
    // Cut the image in order to resize it to the smalled square/rectangle possible
    // To Do ...
    return 1;
}


int initialImageSegmentation(Mat &src,
                             vector<Point> &candidateCentroids,
                             const int superPixelEdge,
                             const ExtractionThresholds thresholds,
                             const RoadOffsets offsets) {
    printThresholds(thresholds);
    printOffsets(offsets);

    preprocessing(src, src, offsets.Horizon_Offset);

    extractPossiblePotholes(src, candidateCentroids, superPixelEdge, thresholds, offsets, " Start image");

    return 1;
}



/***** FEATURE EXTRACTION SEGMENTATION ****/

/*
 * if there isn't a super pixel that is recognized as a pothole the function will return the first superpixel
 * */
SuperPixel selectPothole(Mat src, int nSuperPixels, Mat labels, Mat contour) {
    SuperPixel selected = getSuperPixel(src, 0, labels, contour);
    double averagePixelValue = (double) mean(src)[0];
    //vector<SuperPixel> possiblePotholes = vector<SuperPixel>();
    //Select all possible potholes
    for (int l = 0; l < nSuperPixels; ++l) {
        SuperPixel currSp = getSuperPixel(src, l, labels, contour);
        //Point2d center = calculateSuperPixelCenter(currSp.points);
        if (isPothole(currSp, selected, averagePixelValue)) {
            selected = currSp;
            //possiblePotholes.push_back(currSp);
        }
    }

    /*if(possiblePotholes.size() > 0) {
        sort(possiblePotholes.begin(),
             possiblePotholes.end(),
             [](SuperPixel a, SuperPixel b) -> bool { return (a.meanColourValue[0] < b.meanColourValue[0]); }
        );

        selected = possiblePotholes[possiblePotholes.size() - 1];
    } */


    return selected;
}

/*
 * 1. Segment the candidate with super pixeling
 * 2. Calculate the pixel colour average value
 * In order to isolate the pothole super pixel from car's bumper
 * 3. Select the super pixel that is darker than the average pixel value and lighter than a specified threshold
 * */
SuperPixel extractPotholeRegionFromCandidate(Mat &src, string candidateName) {
    Mat res;
    Mat mask;
    Mat labels;
    Mat contour;
    Ptr<SuperpixelSLIC> superPixels = initSuperPixelingSLIC(src, contour, labels, mask);

    SuperPixel selected = selectPothole(src, superPixels->getNumberOfSuperpixels(), labels, contour);

    //cout << "SP, Size, Area, Variance, Density" << endl;
    return selected;
}