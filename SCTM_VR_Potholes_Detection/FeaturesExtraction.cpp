//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"


typedef struct FeaturesVectors {
    vector<float> averageGreyLevels = vector<float>();
    vector<Mat> histograms = vector<Mat>();
    vector<float> contrasts = vector<float>();
    vector<float> entropies = vector<float>();
    vector<float> skewnesses = vector<float>();
    vector<float> energies = vector<float>();
} FeaturesVectors;


/*
* Each centroid of superpixels extracted in the previous phase
*  1. Candidate will be converted to greyscale
*  2. The pothole region will be segmented another time with super pixeling
*  3. The histogram will be calculated
*  4. Calculate the average gray value
*  5. Calculate the contrast
*  6. Calculate 3-order moments (is Skewness according to http://aishack.in/tutorials/image-moments/)
*  7. Calculate Consistency (After only searches I think that the consistency is what the professor calls energy)
*  8. Calculate Entropy
*  9. Put the calculated features in a eigenvector
*
* */
Features candidateFeatureExtraction(Point centroid, Mat sourceImage, Size candidate_size) {
    //tlc (top left corner) brc(bottom right corner)
    auto tlc_x = centroid.x - candidate_size.width * 0.5;
    auto tlc_y = centroid.y - candidate_size.height * 0.5;
    auto brc_x = centroid.x + candidate_size.width * 0.5;
    auto brc_y = centroid.y + candidate_size.height * 0.5;

    auto tlc = Point2d(tlc_x < 0 ? 0 : tlc_x, tlc_y < 0 ? 0 : tlc_y);
    auto brc = Point2d(brc_x > sourceImage.cols - 1 ? sourceImage.cols : brc_x,
                       brc_y > sourceImage.rows - 1 ? sourceImage.rows : brc_y);

    auto candidate = sourceImage(Rect(tlc, brc));

    //candidates.push_back(candidate);
    Mat candidateGrayScale;
    // Switch color-space from RGB to GreyScale
    cvtColor(candidate, candidateGrayScale, CV_BGR2GRAY);

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

    energy = sqrtf(energy);
    entropy = 0 - entropy;

    float averageGreyValue = (float) mean(candidateGrayScale)[0];
    float skewness = calculateSkewnessGrayImage(candidate, averageGreyValue);

    auto c_name = "Candidate @ (" + to_string(centroid.x) + ", " + to_string(centroid.y) + ")";
    Mat histogram = ExtractHistograms(candidateGrayScale, c_name);

    return Features{candidate, histogram, averageGreyValue, contrast, entropy, skewness, energy};
}

FeaturesVectors normalizeFeatures(double minValue, double maxValue, FeaturesVectors notNormalizedFeatures) {
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


vector<Features> extractFeatures(Mat sourceImage, vector<Point> centroids, Size candidate_size) {

//    auto candidates = vector<Mat>();
    auto notNormalizedfeatures = vector<Features>();
    auto normalizedFeatures = vector<Features>();

    FeaturesVectors candidatesFeaturesVectors;

    /*
        * Each centroid of superpixels extracted in the previous phase
        *  1. Candidate will be converted to greyscale
        *  2. The histogram will be calculated
        *  3. Calculate the average gray value
        *  4. Calculate the contrast
        *  5. Calculate 3-order moments (is Skewness according to http://aishack.in/tutorials/image-moments/)
        *  6. Calculate Consistency (After only searches I think that the consistency is what the professor calls energy)
        *  7. Calculate Entropy
        *  8. Put the calculated features in a eigenvector
        *
        * */
    /*------------------------Candidate Extraction---------------------------*/
    for (auto c : centroids) {
        Features candidateFeatures = candidateFeatureExtraction(c, sourceImage, candidate_size);
        notNormalizedfeatures.push_back(candidateFeatures);

        candidatesFeaturesVectors.histograms.push_back(candidateFeatures.histogram);
        candidatesFeaturesVectors.contrasts.push_back(candidateFeatures.contrast);
        candidatesFeaturesVectors.energies.push_back(candidateFeatures.energy);
        candidatesFeaturesVectors.entropies.push_back(candidateFeatures.entropy);
        candidatesFeaturesVectors.averageGreyLevels.push_back(candidateFeatures.averageGreyValue);
        candidatesFeaturesVectors.skewnesses.push_back(candidateFeatures.skewness);
    }

    /*------------- Normalization Phase -------------*/

    //Normalize feature values to [1,10]
    FeaturesVectors normalizedFeaturesVectors = normalizeFeatures(1, 10, candidatesFeaturesVectors);

    for (int i = 0; i < centroids.size(); i++) {
        /*cout <<
             "Average Gray Value: " << normalizedFeaturesVectors.averageGreyLevels.at(i) <<
             " Contrast: " << normalizedFeaturesVectors.contrasts.at(i) <<
             " Skeweness: " << normalizedFeaturesVectors.skewnesses.at(i) <<
             " Energy: " << normalizedFeaturesVectors.energies.at(i) <<
             " Entropy: " << normalizedFeaturesVectors.entropies.at(i) << endl;*/

        normalizedFeatures.push_back(Features{
                notNormalizedfeatures.at(i).candidate,
                normalizedFeaturesVectors.histograms.at(i),
                normalizedFeaturesVectors.averageGreyLevels.at(i),
                normalizedFeaturesVectors.contrasts.at(i),
                normalizedFeaturesVectors.entropies.at(i),
                normalizedFeaturesVectors.skewnesses.at(i),
                normalizedFeaturesVectors.energies.at(i)
        });
    }

    return normalizedFeatures;
}