//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"
#include "Segmentation.h"
#include "MathUtils.h"
#include "HOG.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv::ximgproc;

typedef struct FeaturesVectors {
    vector<float> averageGreyLevels = vector<float>();
    vector<Mat> histograms = vector<Mat>();
    vector<float> contrasts = vector<float>();
    vector<float> entropies = vector<float>();
    vector<float> skewnesses = vector<float>();
    vector<float> energies = vector<float>();
} FeaturesVectors;

/*
*  Feature extraction from a candidate:
*  1. Candidate will be converted to greyscale
*  2. The pothole region will be segmented another time with super pixeling
*  3. The histogram will be calculated
*  4. Calculate the average gray value
*  5. Calculate the contrast
*  6. Calculate Entropy
*  7. Calculate Energy
*  8. Calculate 3-order moments (is Skewness according to http://aishack.in/tutorials/image-moments/)
* */
cv::Optional<Features> candidateFeatureExtraction(const Mat &src,
                                                  const SuperPixel &nativeSuperPixel,
                                                  const Size &candidateSize,
                                                  const RoadOffsets &offsets,
                                                  const ExtractionThresholds &thresholds) {

    auto centroid = nativeSuperPixel.center;

    // tlc = top left corner brc = bottom right corner
    Point2d tlc = calculateTopLeftCorner(centroid, candidateSize);
    Point2d brc = calculateBottomRightCorner(centroid, src, candidateSize);

    const Mat sample = src(Rect(tlc, brc));
    Mat1b sampleRoadMask = Mat::zeros(sample.rows, sample.cols, CV_8UC1);

    for (int i = 0; i < sample.rows; ++i) {
        for (int j = 0; j < sample.cols; ++j) {

            if (isRoad(src.rows, src.cols, offsets, tlc + Point2d(j, i))) {
                sampleRoadMask.at<uchar>(i, j) = 255;
            }
        }
    }

    auto c_name = "Candidate @ (" + to_string(centroid.x) + ", " + to_string(centroid.y) + ")";

    // 1. Extract only the pothole region
    auto opt = extractPotholeRegionFromCandidate(sample, sampleRoadMask, thresholds);

    if (!opt.hasValue()) return cv::Optional<Features>();

    auto candidateSuperPixel = opt.getValue();

//    imshow("Sample", sample);
//    Mat cnt; sample.copyTo(cnt);
//    cnt.setTo(Scalar(0,0,255), candidateSuperPixel.contour);
//    imshow("Candidate", cnt);
//    waitKey();

    // 2. Switch color-space from RGB to GreyScale
    Mat candidateGrayScale;
    cvtColor(candidateSuperPixel.selection, candidateGrayScale, CV_BGR2GRAY);

    //3. Calculate HOG
    HOG hog;
    hog = calculateHOG(sample, defaultConfig);
    int scaleFactor = 5;
    double viz_factor = 5.0;
    vector<OrientedGradientInCell> greaterOrientedGradientsVector = computeGreaterHOGCells(candidateGrayScale,
                                                                                           hog.descriptors,
                                                                                           defaultConfig.cellSize);
    Mat hogImage = overlapOrientedGradientCellsOnImage(candidateGrayScale,
                                                       greaterOrientedGradientsVector,
                                                       defaultConfig.cellSize,
                                                       scaleFactor,
                                                       viz_factor);


    auto orientedGradientOfTheSuperPixel = selectNeighbourhoodCellsAtContour(candidateSuperPixel.contour,
                                                                             greaterOrientedGradientsVector);

    Mat superPixelHogImage = overlapOrientedGradientCellsOnImage(candidateGrayScale,
                                                                 orientedGradientOfTheSuperPixel,
                                                                 defaultConfig.cellSize,
                                                                 scaleFactor,
                                                                 viz_factor);
//    imshow(c_name + " Hog matrix", hogImage);
//    imshow(c_name + " super pixel Hog matrix", superPixelHogImage);
//    imshow(c_name + " Sample", sample);
//    imshow(c_name + " Superpixel contour", candidateSuperPixel.contour);

    // 4. The histogram will be calculated
//    Mat histogram = ExtractHistograms(candidateGrayScale, c_name);

    // 5. Calculate the average gray value
    float averageGreyValue = (float) mean(candidateGrayScale)[0];

    // 6. Calculate the contrast
    // 7. Calculate Entropy
    // 8. Calculate Energy
    // In order to reduce computation complexity
    // calculate the contrast, entropy and energy with in the same loops
    float contrast = 0.0;
    float entropy = 0.0;
    float energy = 0.0;

//    for (int i = 0; i < candidateGrayScale.rows; i++) {
//        for (int j = 0; j < candidateGrayScale.cols; j++) {
//            contrast = contrast + powf((i - j), 2) * candidateGrayScale.at<uchar>(i, j);
//            entropy = entropy + (candidateGrayScale.at<uchar>(i, j) * log10f(candidateGrayScale.at<uchar>(i, j)));
//            energy = energy + powf(candidateGrayScale.at<uchar>(i, j), 2);
//        }
//    }

    for (Point coordinates : candidateSuperPixel.points) {
        contrast += (coordinates.y - coordinates.x) * (coordinates.y - coordinates.x) *
                    candidateGrayScale.at<uchar>(coordinates);
        entropy += (candidateGrayScale.at<uchar>(coordinates) * log10f(candidateGrayScale.at<uchar>(coordinates)));
        energy += candidateGrayScale.at<uchar>(coordinates) * candidateGrayScale.at<uchar>(coordinates);
    }

    entropy = -entropy;
    energy = sqrtf(energy);

    //9. Calculate Skewness
//    float skewness = calculateSkewnessGrayImage(candidateGrayScale, averageGreyValue);
    float skewness = calculateSkewnessGrayImageRegion(candidateSuperPixel.selection, candidateSuperPixel.points,
                                                      averageGreyValue);

    // Highlights the selected pothole region
//    imshow("Sample " + to_string(nativeSuperPixel.label), sample);
//    waitKey();

    Mat tmp;
    sample.copyTo(tmp);
    tmp.setTo(Scalar(0, 0, 255), candidateSuperPixel.contour);

    return cv::Optional<Features>(Features{
            nativeSuperPixel.label, tmp, Mat(),
            averageGreyValue, contrast, entropy, skewness, energy
    });
}

FeaturesVectors
normalizeFeatures(const double minValue, const double maxValue, const FeaturesVectors &notNormalizedFeatures) {
    FeaturesVectors normalizedFeatures;

    for (auto notNormHistogram : notNormalizedFeatures.histograms) {
        Mat normHistogram;
        normalize(notNormHistogram, normHistogram, minValue, maxValue, NORM_MINMAX, -1, Mat());
        normalizedFeatures.histograms.push_back(normHistogram);
    }

    normalize(notNormalizedFeatures.averageGreyLevels, normalizedFeatures.averageGreyLevels, minValue, maxValue,
              NORM_MINMAX, -1, Mat());
    normalize(notNormalizedFeatures.contrasts, normalizedFeatures.contrasts, minValue, maxValue, NORM_MINMAX, -1,
              Mat());
    normalize(notNormalizedFeatures.entropies, normalizedFeatures.entropies, minValue, maxValue, NORM_MINMAX, -1,
              Mat());
    normalize(notNormalizedFeatures.skewnesses, normalizedFeatures.skewnesses, minValue, maxValue, NORM_MINMAX, -1,
              Mat());
    normalize(notNormalizedFeatures.energies, normalizedFeatures.energies, minValue, maxValue, NORM_MINMAX, -1, Mat());

    return normalizedFeatures;
}


vector<Features> extractFeatures(const Mat &src, const vector<SuperPixel> &candidateSuperPixels,
                                 const Size &candidate_size,
                                 const RoadOffsets &offsets,
                                 const ExtractionThresholds &thresholds) {

//    auto candidates = vector<Mat>();
    auto notNormalizedfeatures = vector<Features>();
    auto normalizedFeatures = vector<Features>();

    FeaturesVectors candidatesFeaturesVectors;

    //Mat imageGrayScale;
    //cvtColor(src, imageGrayScale, CV_BGR2GRAY);
    //Gradient grad = calculateGradient(imageGrayScale);
    //imwrite("../data/gradiente/" + c_name + " gradientey.bmp", resulty);
    //imshow("Gradient module", grad.module);
    //imshow("Gradient x", grad.x);
    //imshow("Gradient y", grad.y);

    /*------------------------Candidate Extraction---------------------------*/
    for (auto candidate : candidateSuperPixels) {

        cv::Optional<Features> optional = candidateFeatureExtraction(src, candidate, candidate_size, offsets,
                                                                     thresholds);

        if (optional.hasValue()) {
            auto candidateFeatures = optional.getValue();
            notNormalizedfeatures.push_back(candidateFeatures);

//        cout << "SP" << candidate.label <<
//             "| AvgGrayVal: " << candidateFeatures.averageGreyValue <<
//             "| Contrast: " << candidateFeatures.contrast <<
//             "| Skeweness: " << candidateFeatures.skewness <<
//             "| Energy: " << candidateFeatures.energy <<
//             "| Entropy: " << candidateFeatures.entropy << endl;

            candidatesFeaturesVectors.histograms.push_back(candidateFeatures.histogram);
            candidatesFeaturesVectors.contrasts.push_back(candidateFeatures.contrast);
            candidatesFeaturesVectors.energies.push_back(candidateFeatures.energy);
            candidatesFeaturesVectors.entropies.push_back(candidateFeatures.entropy);
            candidatesFeaturesVectors.averageGreyLevels.push_back(candidateFeatures.averageGreyValue);
            candidatesFeaturesVectors.skewnesses.push_back(candidateFeatures.skewness);
        }
    }

    /*------------- Normalization Phase -------------*/

    //Normalize feature values to [1,10]
    FeaturesVectors normalizedFeaturesVectors = normalizeFeatures(1.0, 10.0, candidatesFeaturesVectors);

    for (int i = 0; i < notNormalizedfeatures.size(); i++) {
        /*cout <<
             "Average Gray Value: " << normalizedFeaturesVectors.averageGreyLevels.at(i) <<
             " Contrast: " << normalizedFeaturesVectors.contrasts.at(i) <<
             " Skeweness: " << normalizedFeaturesVectors.skewnesses.at(i) <<
             " Energy: " << normalizedFeaturesVectors.energies.at(i) <<
             " Entropy: " << normalizedFeaturesVectors.entropies.at(i) << endl;*/

        normalizedFeatures.push_back(Features{
                notNormalizedfeatures[i].label,
                notNormalizedfeatures[i].candidate,
                normalizedFeaturesVectors.histograms[i],
                normalizedFeaturesVectors.averageGreyLevels[i],
                normalizedFeaturesVectors.contrasts[i],
                normalizedFeaturesVectors.entropies[i],
                normalizedFeaturesVectors.skewnesses[i],
                normalizedFeaturesVectors.energies[i]
        });
    }

    return normalizedFeatures;
}