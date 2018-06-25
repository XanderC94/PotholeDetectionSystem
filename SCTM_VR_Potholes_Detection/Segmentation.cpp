#include <opencv2/photo.hpp>
#include <iostream>
#include "Segmentation.h"
#include "SuperPixelingUtils.h"

using namespace std;

const int RESIZING_WIDTH = 640;
const int RESIZING_HEIGHT = 480;

const int MAX_AREA = RESIZING_HEIGHT * RESIZING_WIDTH;


void preprocessing(Mat &src, Mat &processedImage, const double Horizon_Offset) {

    float aspectRatio = static_cast<float>(src.cols) / static_cast<float>(src.rows);

    float newHeight = sqrtf(MAX_AREA / aspectRatio);
    float newWidth = newHeight * aspectRatio;

    Size scale(static_cast<int>(newWidth), static_cast<int>(newHeight));

    resize(src, processedImage, scale);

    // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
    GaussianBlur(processedImage, processedImage, Size(3, 3), 0.0); // OK, do not change

    processedImage = processedImage(
            Rect(Point2d(0, newHeight * Horizon_Offset), Point2d(newWidth - 1, newHeight - 1)));
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

    double ratioDark[3] = {0.0, 0.0, 0.0};

    ratioDark[0] = neighborsMeanColourValue.val[0] / superPixel.meanColourValue.val[0];
    ratioDark[1] = neighborsMeanColourValue.val[1] / superPixel.meanColourValue.val[1];
    ratioDark[2] = neighborsMeanColourValue.val[2] / superPixel.meanColourValue.val[2];

//    if ((ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.colourRatioThresholdMin &&
//        (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.colourRatioThresholdMax && src.rows <= 128) {
//
//        Mat tmp; src.copyTo(tmp, selectionMask);
//        tmp.setTo(Scalar(0, 0, 255), (labels == superPixel.label));
//        imshow("TMask " + to_string(superPixel.label), tmp);
//        waitKey();
//
//        cout << "SP n° " << superPixel.label
//             << " \t| Ratio: [" << ratioDark[0] << ", " << ratioDark[1] << ", " << ratioDark[2] << "]"
//             << endl;
//    }

    return (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.colourRatioThresholdMin &&
           (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.colourRatioThresholdMax;
}

set<int> findNeighbors(const Point &candidate, const Mat &labels, const int edge) {

    vector<Point> testPoints = {
            candidate + Point(edge, 0),         // Up
            candidate + Point(-edge, 0),        // Down
            candidate + Point(0, edge),         // Right
            candidate + Point(0, -edge),        // Left
            candidate + Point(edge, edge),      // Up Right
            candidate + Point(-edge, -edge),    // Up Left
            candidate + Point(-edge, edge),     // Down Right
            candidate + Point(edge, -edge)      // Down Left
    };

    vector<Point> stepPoints = {
            Point(edge, 0),         // Up
            Point(-edge, 0),        // Down
            Point(0, edge),         // Right
            Point(0, -edge),        // Left
            Point(edge, edge),      // Up Right
            Point(-edge, -edge),    // Up Left
            Point(-edge, edge),     // Down Right
            Point(edge, -edge)      // Down Left
    };

    set<int> neighborhood;

    for (int i = 0; i < testPoints.size(); ++i) {
        auto neighbor = testPoints[i];
        auto step = stepPoints[i];
        bool found = false;
        // Check if the pixel is inside the image boundaries
        while ((neighbor.y > 0 && neighbor.x > 0 && neighbor.y < labels.rows - 1 && neighbor.x < labels.cols - 1)
               && !found) {
            // Get the label of the neighbor
            int n = labels.at<int>(neighbor);

            if (n != labels.at<int>(candidate)) {
                neighborhood.insert(labels.at<int>(neighbor));
                found = true;
            } else {
                neighbor += step;
            }
        }
    }

    return neighborhood;
}

int extractRegionsOfInterest(const Ptr<SuperpixelLSC> &superPixeler,
                             const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                             const int superPixelEdge,
                             const ExtractionThresholds thresholds,
                             const RoadOffsets offsets) {

    Mat contour, mask, labels, meanColourMask;

    superPixeler->iterate(10);
//    superPixeler->getLabelContourMask(contour);
    superPixeler->getLabels(labels);

//    src.copyTo(mask);
//    mask.setTo(Scalar(255, 255, 255));
//    src.copyTo(meanColourMask);

    for (int superPixelLabel = 0; superPixelLabel < superPixeler->getNumberOfSuperpixels(); ++superPixelLabel) {

        SuperPixel superPixel = getSuperPixel(src, superPixelLabel, labels, offsets);

//        Scalar color_mask_value = Scalar(0, 0, 0);

        if (isRoad(src.rows, src.cols, offsets, superPixel.center)
            && calculateSuperPixelDensity(superPixel.points) < thresholds.Density_Threshold
            && superPixel.points.size() > 256) {

            superPixel.neighbors = findNeighbors(superPixel.center, labels, superPixelEdge / 4);

            if (isSuperpixelOfInterest(src, labels, superPixel, thresholds)) {
//                color_mask_value = Scalar(255, 255, 255);
                // Add the superpixel to the candidates vector
                candidateSuperpixels.push_back(superPixel);
            }
        }

//        meanColourMask.setTo(superPixel.meanColourValue, superPixel.mask);
//        mask.setTo(color_mask_value, superPixel.mask);

    }

//    imshow("Src", src);
//
//    meanColourMask.setTo(Scalar(0, 0, 255), contour);
//    imshow("Segmentation", meanColourMask);
//
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
SuperPixel selectPothole(const Mat &src, const int nSuperPixels, const Mat &labels) {

    SuperPixel selected = getSuperPixel(src, 0, labels);
    double averagePixelValue = (double) mean(src)[0];
    // vector<SuperPixel> possiblePotholes = vector<SuperPixel>();
    //Select all possible potholes
    for (int l = 0; l < nSuperPixels; ++l) {
        SuperPixel currSp = getSuperPixel(src, l, labels);
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
std::optional<SuperPixel> extractPotholeRegionFromCandidate(const Ptr<SuperpixelLSC> superPixeler, const Mat &src,
                                                            const ExtractionThresholds thresholds) {
    Mat res, labels, contours;
    vector<SuperPixel> soi;
    SuperPixel product;

    superPixeler->iterate(10);
    superPixeler->getLabels(labels);

    for (int superPixelLabel = 0; superPixelLabel < superPixeler->getNumberOfSuperpixels(); ++superPixelLabel) {

        SuperPixel superPixel = getSuperPixel(src, superPixelLabel, labels);

        superPixel.neighbors = findNeighbors(superPixel.center, labels, src.rows / 8);

        if (isSuperpixelOfInterest(src, labels, superPixel, thresholds)) {
            // Add the superpixel to the candidates vector

            soi.push_back(superPixel);
        }
    }

    // Join the detected regions
    if (!soi.empty()) {
        product = soi[0];
        if (soi.size() > 1) {
            for (int i = 1; i < soi.size(); ++i) {
                product.label += soi[i].label;
                product.mask += soi[i].mask;
                product.neighbors.merge(soi[i].neighbors);
                product.meanColourValue = (product.meanColourValue + soi[i].meanColourValue) * 0.5;
                product.points.insert(product.points.end(), soi[i].points.begin(), soi[i].points.end());
            }
            product.center = calculateSuperPixelCenter(product.points);
            product.contour = getContours(product.mask);

            src.copyTo(product.selection, product.mask);
        }

//        src.copyTo(res);
//        res.setTo(Scalar(0,0,255), product.contour);
//        imshow("S", res);
//        imshow("M", product.mask);
//        waitKey();

        return optional(product);
    }

    return optional<SuperPixel>();
}