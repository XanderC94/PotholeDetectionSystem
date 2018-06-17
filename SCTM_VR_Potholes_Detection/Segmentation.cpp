#include "Segmentation.h"
#include "MathUtils.h"

#include <iostream>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

const int RESIZING_WIDTH = 640;
const int RESIZING_HEIGHT = 480;


void preprocessing(Mat &src, Mat &processedImage, const double Horizon_Offset) {

    Size scale(RESIZING_WIDTH, RESIZING_HEIGHT);

    resize(src, processedImage, scale);

    // Delete Reflection Noises
//    fastNlMeansDenoisingColored(processedImage, processedImage, 3.0, 10.0, 7, 21);

    // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
    GaussianBlur(processedImage, processedImage, Size(3, 3), 0.0); // OK, do not change

    processedImage = processedImage(
            Rect(Point2d(0, RESIZING_HEIGHT * Horizon_Offset), Point2d(RESIZING_WIDTH - 1, RESIZING_HEIGHT - 1)));

}

bool isRoad(const int H, const int W, const RoadOffsets offsets, const Point2d center) {
    // How to evaluate offsets in order to separate road super pixels?
    // Or directly identify RoI (Region of interest) where a pothole will be more likely detected?
    // ...
    // Possibilities
    // => Gaussian 3D function
    // => Evaluates the pixels through an Analytic Rect Function : F(x) > 0 is over the rect, F(x) < 0 is under, F(x) = 0 it lies on.

    bool isRoad = AnalyticRect2D(cv::Point2d(W * offsets.SLine_X_Left_Offset, H * offsets.SLine_Y_Offset),
                                 cv::Point2d(W * 0.4, 0.0), center) >= 0 &&
                  AnalyticRect2D(
                          cv::Point2d(W * (1.0 - offsets.SLine_X_Right_Offset), H * offsets.SLine_Y_Offset),
                          cv::Point2d(W * 0.6, 0.0), center) >= 0;

    return isRoad;
}

/*
 * Check if the super pixel is a pothole:
 *      - checking if its density il less than the specified density threshold
 *      - checking if the variance on x axis or y axis is greater than the variance threshold
 * */
bool isSuperpixelOfInterest(const Mat &src, const Mat &labels, const SuperPixel &superPixel,
                            ExtractionThresholds thresholds) {

    Mat1b selectionMask = (labels == -1);

    for (int n : superPixel.neighbors) selectionMask.setTo(Scalar(255), (labels == n));

    auto neighborsMeanColourValue = mean(src, selectionMask);

    double ratioDark[3] {0.0, 0.0, 0.0};

    ratioDark[0] = neighborsMeanColourValue.val[0] / superPixel.meanColourValue.val[0];
    ratioDark[1] = neighborsMeanColourValue.val[1] / superPixel.meanColourValue.val[1];
    ratioDark[2] = neighborsMeanColourValue.val[2] / superPixel.meanColourValue.val[2];

//    Point2d deviation =  calculateSuperPixelVariance(superPixel.points, superPixel.center);

    double density = calculateSuperPixelDensity(superPixel.points);

    if ((ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.colourRatioThresholdMin &&
        (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.colourRatioThresholdMax) {

//        Mat tmp; src.copyTo(tmp, selectionMask);
//        tmp.setTo(Scalar(0, 0, 255), (labels == superPixel.label));
//        imshow("TMask " + to_string(superPixel.label), tmp);
//        waitKey();

        cout << "SP n° " << superPixel.label
//             << " \t| Mean: " << superPixel.meanColourValue
//             << " \t| NeighborsMean: " << neighborsMeanColourValue
             << " \t| Ratio: [" << ratioDark[0] << ", " << ratioDark[1] << ", " << ratioDark[2] << "]"
             << " \t| Density:" << density
//             << " \t| Deviation:" << deviation
             << endl;
    }

    return density < thresholds.Density_Threshold &&
//            (deviation.x > thresholds.Variance_Threshold || deviation.y > thresholds.Variance_Threshold) &&
            (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.colourRatioThresholdMin &&
            (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.colourRatioThresholdMax;
}

set<int> findNeighbors(const Point &candidate, const Mat &labels, const int edge) {

    vector<Point> testPoints = {
            candidate + Point(edge, 0),       // Up
            candidate - Point(edge, 0),       // Down
            candidate + Point(0, edge),       // Right
            candidate - Point(0, edge),       // Left
            candidate + Point(edge, edge),    // Up Right
            candidate - Point(edge, edge),    // Up Left
            candidate + Point(-edge, edge),   // Down Right
            candidate + Point(edge, -edge)    // Down Left
    };

    set<int> neighborhood;

    for (Point &neighbor : testPoints) {
        // Check if the pixel is inside the image boundaries
        if (neighbor.y > 0 && neighbor.x > 0 && neighbor.y < labels.rows - 1 && neighbor.x < labels.cols - 1) {
            // Get the label of the neighbor
            int n = labels.at<int>(neighbor);
            if (n > 0 && n != labels.at<int>(candidate)) {
                neighborhood.insert(labels.at<int>(neighbor));
            }
        }
    }

    return neighborhood;
}

int extractRegionsOfInterest(const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                             const int superPixelEdge,
                             const ExtractionThresholds thresholds,
                             const RoadOffsets offsets) {

    Mat contour, mask, labels, meanColourMask;

    Ptr<SuperpixelLSC> superpixels = initSuperPixelingLSC(src, contour, mask, labels, superPixelEdge);

    src.copyTo(meanColourMask);

    for (int superPixelLabel = 0; superPixelLabel < superpixels->getNumberOfSuperpixels(); ++superPixelLabel) {

        SuperPixel superPixel = getSuperPixel(src, superPixelLabel, labels, contour);

        Scalar color_mask_value = Scalar(0, 0, 0);

        if (isRoad(src.rows, src.cols, offsets, superPixel.center)) {

            superPixel.neighbors = findNeighbors(superPixel.center, labels, superPixelEdge);

            if (isSuperpixelOfInterest(src, labels, superPixel, thresholds)) {
                color_mask_value = Scalar(255, 255, 255);
                // Add the superpixel to the candidates vector
                candidateSuperpixels.push_back(superPixel);
            }
        }

        meanColourMask.setTo(superPixel.meanColourValue, superPixel.selectionMask);
        mask.setTo(color_mask_value, superPixel.selectionMask);
    }

//    imshow("Src", src);

//    out.setTo(Scalar(0, 0, 255), contour);
//    imshow("Segmentation", out);

//    Mat res;
//    src.copyTo(res, mask);
//    res.setTo(Scalar(0,0,255), contour);
//
//    imshow("Result", res);
//
//    waitKey();

    return 1;
}

/***** FEATURE EXTRACTION SEGMENTATION ****/

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