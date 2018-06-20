//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"
#include "Segmentation.h"
#include "HOG.h"

using namespace cv;
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
Features candidateFeatureExtraction(const SuperPixel nativeSuperPixel, const Mat &src, const Size candidate_size) {

    auto centroid = nativeSuperPixel.center;

    // tlc = top left corner brc = bottom right corner
    auto tlc = calculateTopLeftCorner(centroid, candidate_size);
    auto brc = calculateBottomRightCorner(centroid, src, candidate_size);

    Mat sample = src(Rect(tlc, brc));
//    sample.setTo(Scalar(0,0,255), nativeSuperPixel.contour);

    auto c_name = "Candidate @ (" + to_string(centroid.x) + ", " + to_string(centroid.y) + ")";

    // 1. Extract only the pothole region
    Mat candidateForSuperPixeling;
//    cvtColor(candidateGrayScale, candidateForSuperPixeling, CV_GRAY2BGR);
//    SuperPixel candidateSuperPixel = extractPotholeRegionFromCandidate(candidateForSuperPixeling, c_name);
    // 2. Switch color-space from RGB to GreyScale
    Mat candidateGrayScale;
    cvtColor(sample, candidateGrayScale, CV_BGR2GRAY);

    //2. Calculate HoG
    HoG hog;
    hog = calculateHoG(sample, defaultConfig);
    Mat hogImage = getHoGDescriptorVisualImage(candidateGrayScale,
                                               hog.descriptors,
                                               Size(candidateGrayScale.cols, candidateGrayScale.rows),
                                               defaultConfig.cellSize,
                                               5,
                                               2.0);
    imshow(c_name + " Hog matrix", hogImage);


    // 3. The histogram will be calculated
    Mat histogram = ExtractHistograms(candidateGrayScale, c_name);

    //4. Calculate the average gray value
    float averageGreyValue = (float) mean(candidateGrayScale)[0];

    // 5. Calculate the contrast
    // 6. Calculate Entropy
    // 7. Calculate Energy
    // In order to reduce computation complexity
    // calculate the contrast, entropy and energy with in the same loops
    float contrast = 0.0;
    float entropy = 0.0;
    float energy = 0.0;

    for (int i = 0; i < candidateGrayScale.rows; i++) {
        for (int j = 0; j < candidateGrayScale.cols; j++) {
            contrast = contrast + powf((i - j), 2) * candidateGrayScale.at<uchar>(i, j);
            entropy = entropy + (candidateGrayScale.at<uchar>(i, j) * log10f(candidateGrayScale.at<uchar>(i, j)));
            energy = energy + powf(candidateGrayScale.at<uchar>(i, j), 2);
        }
    }

//    for (Point coordinates : candidateSuperPixel.points) {
//        contrast = contrast + (coordinates.y - coordinates.x) * (coordinates.y - coordinates.x) * candidateGrayScale.at<uchar>(coordinates);
//        entropy = entropy +
//                  (candidateGrayScale.at<uchar>(coordinates) * log10f(candidateGrayScale.at<uchar>(coordinates)));
//        energy = energy + candidateGrayScale.at<uchar>(coordinates) * candidateGrayScale.at<uchar>(coordinates);
//    }

    entropy = 0 - entropy;
    energy = sqrtf(energy);

    //8. Calculate Skewness
    float skewness = calculateSkewnessGrayImage(sample, averageGreyValue);
//    float skewness = calculateSkewnessGrayImageRegion(nativeSuperPixel.selection, nativeSuperPixel.points, averageGreyValue);

    // Highlights the selected pothole region
//    imshow("Sample " + to_string(nativeSuperPixel.label), sample);
//    waitKey();

    return Features {
          -1,  sample, histogram, averageGreyValue, contrast, entropy, skewness, energy
    };
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


vector<Features>
extractFeatures(const Mat &src, const vector<SuperPixel> &candidateSuperPixels, const Size candidate_size) {

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
        Features candidateFeatures = candidateFeatureExtraction(candidate, src, candidate_size);
        notNormalizedfeatures.push_back(candidateFeatures);

//        cout << "SP" << candidate.SPLabel <<
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

    /*------------- Normalization Phase -------------*/

    //Normalize feature values to [1,10]
    FeaturesVectors normalizedFeaturesVectors = normalizeFeatures(1.0, 10.0, candidatesFeaturesVectors);

    for (int i = 0; i < candidateSuperPixels.size(); i++) {
        /*cout <<
             "Average Gray Value: " << normalizedFeaturesVectors.averageGreyLevels.at(i) <<
             " Contrast: " << normalizedFeaturesVectors.contrasts.at(i) <<
             " Skeweness: " << normalizedFeaturesVectors.skewnesses.at(i) <<
             " Energy: " << normalizedFeaturesVectors.energies.at(i) <<
             " Entropy: " << normalizedFeaturesVectors.entropies.at(i) << endl;*/

        normalizedFeatures.push_back(Features{
                candidateSuperPixels[i].label,
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