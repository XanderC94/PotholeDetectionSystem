//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"
#include "Segmentation.h"
#include "MathUtils.h"
#include "HistogramElaboration.h"
#include "FeaturesExtractionUtils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv::ximgproc;

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
std::vector<Features> candidateFeatureExtraction(const Mat &src,
                                                 const SuperPixel &nativeSuperPixel,
                                                 const Size &candidateSize,
                                                 const RoadOffsets &offsets,
                                                 const ExtractionThresholds &thresholds) {

    auto centroid = nativeSuperPixel.center;
    vector<Features> candidatesFeatures;
    // tlc = top left corner brc = bottom right corner
    Point2d tlc = calculateTopLeftCorner(centroid, candidateSize);
    Point2d brc = calculateBottomRightCorner(centroid, src, candidateSize);

    const Mat sample = src(Rect(tlc, brc));
    Mat1b exclusionMask = createExclusionMask(src, sample, tlc, offsets, thresholds);

//    auto c_name = "Candidate @ (" + to_string(centroid.x) + ", " + to_string(centroid.y) + ")";

    // 1. Extract only the pothole region
    auto soi = extractPotholeRegionFromCandidate(sample, exclusionMask, thresholds);

    for (int i = 0; i < soi.size(); ++i) {
        const auto &candidateSuperPixel = soi[i];
        // 2. Switch color-space from BGR to GreyScale
        Mat candidateGrayScale, sampleGS;
        cvtColor(candidateSuperPixel.selection, candidateGrayScale, CV_BGR2GRAY);

        //3. Calculate HOG
        Mat1f hogParams = extractHogParams(sample, candidateGrayScale, candidateSuperPixel);

        // 4. The histogram will be calculated
        Mat histogram = ExtractHistograms(candidateGrayScale, candidateSuperPixel.mask, 256);

        // 5. Calculate the average gray value
        float averageGreyValue = (float) mean(candidateGrayScale, candidateSuperPixel.mask)[0];
        Scalar averageRGBValues = mean(candidateSuperPixel.selection, candidateSuperPixel.mask);

        // 6. Calculate the contrast
        // 7. Calculate Entropy
        // 8. Calculate Energy

        float contrast = 0.0;
        float entropy = 0.0;
        float energy = 0.0;
        calculateContrastEntropyEnergy(contrast, entropy, energy, candidateSuperPixel, candidateGrayScale);

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

        candidatesFeatures.push_back(Features{
                ._class = -1,
                .label = nativeSuperPixel.label,
                .id = i,
                .candidate = tmp,
                .histogram = histogram,
                .averageGreyValue = averageGreyValue,
                .averageRGBValues = averageRGBValues,
                .contrast = contrast,
                .entropy = entropy,
                .skewness = skewness,
                .energy = energy,
                .hogDescriptors = hogParams
        });
    }

    return candidatesFeatures;
}

FeaturesVectors
normalizeFeatures(const double minValue, const double maxValue, const FeaturesVectors &notNormalizedFeatures) {
    FeaturesVectors normalizedFeatures;

    for (const auto &notNormHistogram : notNormalizedFeatures.histograms) {
        Mat normHistogram;
        normalize(notNormHistogram, normHistogram, minValue, maxValue, NORM_MINMAX, -1, Mat());
        normalizedFeatures.histograms.push_back(normHistogram);
    }

    normalize(notNormalizedFeatures.averageGreyLevels, normalizedFeatures.averageGreyLevels, minValue, maxValue,
              NORM_MINMAX, -1, Mat());
    normalize(notNormalizedFeatures.averageRedLevels, normalizedFeatures.averageRedLevels, minValue, maxValue,
              NORM_MINMAX, -1, Mat());
    normalize(notNormalizedFeatures.averageGreenLevels, normalizedFeatures.averageGreenLevels, minValue, maxValue,
              NORM_MINMAX, -1, Mat());
    normalize(notNormalizedFeatures.averageBlueLevels, normalizedFeatures.averageBlueLevels, minValue, maxValue,
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

    /*------------------------Candidate Extraction---------------------------*/
    for (const auto &candidate : candidateSuperPixels) {

        auto spFeatures = candidateFeatureExtraction(src, candidate, candidate_size, offsets,
                                                                     thresholds);

        if (!spFeatures.empty()) {
            for (const auto &candidateFeatures : spFeatures) {

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
                candidatesFeaturesVectors.averageRedLevels.push_back(
                        static_cast<float>(candidateFeatures.averageRGBValues.val[2]));
                candidatesFeaturesVectors.averageGreenLevels.push_back(
                        static_cast<float>(candidateFeatures.averageRGBValues.val[1]));
                candidatesFeaturesVectors.averageBlueLevels.push_back(
                        static_cast<float>(candidateFeatures.averageRGBValues.val[0]));
                candidatesFeaturesVectors.skewnesses.push_back(candidateFeatures.skewness);
            }
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
                notNormalizedfeatures[i]._class,
                notNormalizedfeatures[i].label,
                notNormalizedfeatures[i].id,
                notNormalizedfeatures[i].candidate,
                normalizedFeaturesVectors.histograms[i],
                normalizedFeaturesVectors.averageGreyLevels[i],
                Scalar(
                        normalizedFeaturesVectors.averageRedLevels[i],
                        normalizedFeaturesVectors.averageGreenLevels[i],
                        normalizedFeaturesVectors.averageBlueLevels[i]
                ),
                normalizedFeaturesVectors.contrasts[i],
                normalizedFeaturesVectors.entropies[i],
                normalizedFeaturesVectors.skewnesses[i],
                normalizedFeaturesVectors.energies[i],
                notNormalizedfeatures[i].hogDescriptors

        });
    }

    return normalizedFeatures;
}