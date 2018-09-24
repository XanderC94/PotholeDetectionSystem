#include <opencv2/photo.hpp>
#include <iostream>
#include "../include/phdetection/segmentation.hpp"
#include "../include/phdetection/io.hpp"

using namespace cv;
using namespace cv::ximgproc;
using namespace phd::ontologies;
using namespace std;
using namespace phd::superpixeling;

namespace phd::segmentation {

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
 *      - comparing if the ratio between the mean neigbour superpixels color and the current superpixel colour
 *      is between a specified range.
 * */
    bool isSuperpixelOfInterest(const Mat &src, const Mat &labels, const SuperPixel &superPixel,
                                ExtractionThresholds thresholds) {

        Mat1b selectionMask = (labels == -1);

        for (int n : superPixel.neighbors) selectionMask.setTo(Scalar(255), (labels == n));

        auto neighborsMeanColourValue = mean(src, selectionMask);

        double ratioDark[3] = {0.0, 0.0, 0.0};

        ratioDark[0] = neighborsMeanColourValue.val[0] / superPixel.meanColour.val[0];
        ratioDark[1] = neighborsMeanColourValue.val[1] / superPixel.meanColour.val[1];
        ratioDark[2] = neighborsMeanColourValue.val[2] / superPixel.meanColour.val[2];

//    if ((ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.minGreyRatio &&
//        (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.maxGreyRatio) {
//
//        Mat tmp; src.copyTo(tmp, selectionMask);
//        tmp.setTo(Scalar(0, 0, 255), (labels == superPixel.label));
//        imshow("TMask " + to_string(superPixel.label), tmp);
//        waitKey();
//
//        cout << "SP n° " << superPixel.label
//             << "\t| Mean color: " << superPixel.meanColour
//             << " \t| Density: " << calculateSuperPixelDensity(superPixel.points)
//             << " \t| Ratio: [" << ratioDark[0] << ", " << ratioDark[1] << ", " << ratioDark[2] << "]"
//             << endl;
//    }

        return (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 > thresholds.minGreyRatio &&
               (ratioDark[0] + ratioDark[1] + ratioDark[2]) / 3 < thresholds.maxGreyRatio;
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

    void showSPFounded(const Mat src, const Mat contour){
        Mat spContourOnSrc;
        src.copyTo(spContourOnSrc);
        spContourOnSrc.setTo(Scalar(0,0,255), contour);
        phd::io::showElaborationStatusToTheUser("SuperPixel founded", spContourOnSrc);
    }

    void showSPSegmentation(const Mat meanColourMask, const Mat contour){
        Mat tmpMeanColourMask;
        meanColourMask.copyTo(tmpMeanColourMask);
        tmpMeanColourMask.setTo(Scalar(0, 0, 255), contour);
        phd::io::showElaborationStatusToTheUser("Superpixeling Segmentation", tmpMeanColourMask);
    }

    void showSPResult(const Mat src, const Mat mask, const Mat contour){
        Mat res;
        src.copyTo(res, mask);
        res.setTo(Scalar(0,0,255), contour);
        phd::io::showElaborationStatusToTheUser("Superpixeling Result", res);
    }

    void showSuperpixelingElaboration(const Ptr<SuperpixelLSC> &superPixeler,
                                      const Mat src,
                                      const Mat mask,
                                      const Mat meanColourMask){
        Mat contour;
        superPixeler->getLabelContourMask(contour);

        phd::io::showElaborationStatusToTheUser("Segmentation source", src);

        showSPFounded(src, contour);

        showSPSegmentation(meanColourMask, contour);

        showSPResult(src, mask, contour);
    }

    void showSuperpixelingElaboration(const Ptr<SuperpixelLSC> &superPixeler,
                                      const Mat src,
                                      const Mat mask){
        Mat contour;
        superPixeler->getLabelContourMask(contour);

//    showElaborationStatusToTheUser("Segmentation source", src);

        showSPFounded(src, contour);

//    showSPResult(src, mask, contour);
    }

    int extractRegionsOfInterest(const Ptr<SuperpixelLSC> &superPixeler,
                                 const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                                 const int superPixelEdge,
                                 const ExtractionThresholds thresholds,
                                 const RoadOffsets offsets) {

//    Mat contour;
        Mat mask;
        Mat labels;
        Mat meanColourMask;

        superPixeler->iterate(10);
        superPixeler->getLabels(labels);

        src.copyTo(mask);
        mask.setTo(Scalar(255, 255, 255));
        src.copyTo(meanColourMask);

        for (int superPixelLabel = 0; superPixelLabel < superPixeler->getNumberOfSuperpixels(); ++superPixelLabel) {

            SuperPixel superPixel = getSuperPixel(src, superPixelLabel, labels);

            Scalar color_mask_value = Scalar(0, 0, 0);

            if (isRoad(src.rows, src.cols, offsets, superPixel.center)
                && calculateSuperPixelDensity(superPixel.points) < thresholds.density
                && superPixel.points.size() > 256) {

                int neighbourEdge = superPixelEdge / 2; // before was 4
                superPixel.neighbors = findNeighbors(superPixel.center, labels, neighbourEdge);

                if (isSuperpixelOfInterest(src, labels, superPixel, thresholds)) {
                    color_mask_value = Scalar(255, 255, 255);
                    // Add the superpixel to the candidates vector
                    candidateSuperpixels.push_back(superPixel);
                }
            }

            meanColourMask.setTo(superPixel.meanColour, superPixel.mask);
            mask.setTo(color_mask_value, superPixel.mask);

        }

//    showSuperpixelingElaboration(superPixeler, src, mask, meanColourMask);

        return 1;
    }

/***** FEATURE EXTRACTION SEGMENTATION ****/

/*
 * 1. Segment the candidate with super pixeling
 * 2. Calculate the pixel colour average value
 * In order to isolate the pothole super pixel from car's bumper
 * 3. Select the super pixel that is darker than the average pixel value and lighter than a specified threshold
 * */
    std::vector<SuperPixel> extractPotholeRegionFromCandidate(const Mat &candidate,
                                                              const Mat1b &exclusionMask,
                                                              const ExtractionThresholds &thresholds) {
        Mat res;
        Mat labels;
        Mat mask;
        vector<SuperPixel> soi;
        vector<SuperPixel> products;

        Ptr<SuperpixelLSC> superPixeler = initSuperPixelingLSC(candidate, 32); // 32 or 48 are OK

        superPixeler->iterate(10);
        superPixeler->getLabels(labels);

        for (int superPixelLabel = 0; superPixelLabel < superPixeler->getNumberOfSuperpixels(); ++superPixelLabel) {

            SuperPixel superPixel = getSuperPixel(candidate, exclusionMask, superPixelLabel, labels);
            Scalar color_mask_value = Scalar(0, 0, 0);

            int neighbourEdge = candidate.rows / 2;
            superPixel.neighbors = findNeighbors(superPixel.center, labels, neighbourEdge);

            if (isSuperpixelOfInterest(candidate, labels, superPixel, thresholds)) {
                color_mask_value = Scalar(255, 255, 255);
                // Add the superpixel to the candidates vector
                soi.push_back(superPixel);
            }

            mask.setTo(color_mask_value, superPixel.mask);
        }

        auto el = getStructuringElement(MORPH_ELLIPSE, Point(5, 5));

        std::set<int> visited;
        // Join the detected regions
        if (!soi.empty()) {
            // Cycle each superpixel of interest
            for (auto sp : soi) {
                // If not already visited (ndr. found as a neighbor of a previous sp)
                if (visited.count(sp.label) == 0) {
                    SuperPixel product;
                    product.mask = sp.mask;

                    // Check all the others soi
                    for (auto n : soi) {
                        // All those that are neighbors the selected sp are merged with it
                        if (sp.neighbors.count(n.label) == 1) {
                            product.mask += n.mask;
                            visited.insert(n.label);
                        }
                    }

                    dilate(product.mask, product.mask, el);
                    candidate.copyTo(product.selection, product.mask);
                    product.contour = getContoursMask(product.mask);
                    findNonZero(product.mask, product.points);
                    product.center = calculateSuperPixelCenter(product.points);
                    product.meanColour = mean(product.selection, product.mask);

                    auto mean_sum =
                            product.meanColour.val[0] +
                            product.meanColour.val[1] +
                            product.meanColour.val[2];

                    std::cout << "SoI size:" << product.points.size() << endl;

                    if (product.points.size() > 256
                        && static_cast<float>(product.points.size()) / (candidate.rows * candidate.cols) < 0.7f
                        && (mean_sum > 35 * 3 && mean_sum < 225 * 3)) {
                        products.push_back(product);
                    }

                    visited.insert(sp.label);
                }
            }
        }

//    showSuperpixelingElaboration(superPixeler, candidate, mask);
//    showElaborationStatusToTheUser(products);

        return products;
    }
}