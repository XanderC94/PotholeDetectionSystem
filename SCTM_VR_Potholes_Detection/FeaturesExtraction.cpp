//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"

vector<Features> extractFeatures(Mat sourceImage, vector<Point> centroids, Size candidate_size) {

    auto candidates = vector<Mat>();
    auto notNormalizedfeatures = vector<Features>();
    auto normalizedFeatures = vector<Features>();

    auto candidatesAverageGreyLevel = vector<float>();
    auto candidatesContrast = vector<float>();
    auto candidatesEntropy = vector<float>();
    auto candidatesSkewness = vector<float>();
    auto candidatesEnergy = vector<float>();


    Mat candidateGrayScale;

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
        //tlc (top left corner) brc(bottom right corner)
        auto tlc_x = c.x - candidate_size.width * 0.5;
        auto tlc_y = c.y - candidate_size.height * 0.5;
        auto brc_x = c.x + candidate_size.width * 0.5;
        auto brc_y = c.y + candidate_size.height * 0.5;

        auto tlc = Point2d(tlc_x < 0 ? 0 : tlc_x, tlc_y < 0 ? 0 : tlc_y);
        auto brc = Point2d(brc_x > sourceImage.cols - 1 ? sourceImage.cols : brc_x,
                           brc_y > sourceImage.rows - 1 ? sourceImage.rows : brc_y);

        auto candidate = sourceImage(Rect(tlc, brc));

        candidates.push_back(candidate);

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

        candidatesContrast.push_back(contrast);

        energy = sqrtf(energy);
        candidatesEnergy.push_back(energy);

        entropy = 0 - entropy;
        candidatesEntropy.push_back(entropy);

        float averageGreyValue = (float) mean(candidateGrayScale)[0];
        candidatesAverageGreyLevel.push_back(averageGreyValue);

        float skewness = calculateSkewnessGrayImage(candidate, averageGreyValue);
        candidatesSkewness.push_back(skewness);


        auto c_name = "Candidate @ (" + to_string(c.x) + ", " + to_string(c.y) + ")";
        imshow(c_name, candidateGrayScale);

        cout << c_name << endl;
        cout <<
             "Average Gray Value: " << averageGreyValue <<
             " Contrast: " << contrast <<
             " Skeweness: " << skewness <<
             " Energy: " << energy <<
             " Entropy: " << entropy << endl;

        Mat histogram = ExtractHistograms(candidateGrayScale, c_name);

        notNormalizedfeatures.push_back(Features{c, histogram, averageGreyValue, contrast, entropy, skewness, energy});
    }

    /*------------- Normalization Phase -------------*/

    auto normCandidatesAverageGreyLevel = vector<float>();
    auto normCandidatesContrast = vector<float>();
    auto normCandidatesEntropy = vector<float>();
    auto normCandidatesSkewness = vector<float>();
    auto normCandidatesEnergy = vector<float>();

    //Normalize feature values to [1,10]
    normalize(candidatesAverageGreyLevel, normCandidatesAverageGreyLevel, 1, 10, NORM_MINMAX, -1, Mat());
    normalize(candidatesContrast, normCandidatesContrast, 1, 10, NORM_MINMAX, -1, Mat());
    normalize(candidatesEntropy, normCandidatesEntropy, 1, 10, NORM_MINMAX, -1, Mat());
    normalize(candidatesSkewness, normCandidatesSkewness, 1, 10, NORM_MINMAX, -1, Mat());
    normalize(candidatesEnergy, normCandidatesEnergy, 1, 10, NORM_MINMAX, -1, Mat());

    for (int i = 0; i < centroids.size(); i++) {
        cout <<
             "Average Gray Value: " << normCandidatesAverageGreyLevel.at(i) <<
             " Contrast: " << normCandidatesContrast.at(i) <<
             " Skeweness: " << normCandidatesSkewness.at(i) <<
             " Energy: " << normCandidatesEnergy.at(i) <<
             " Entropy: " << normCandidatesEntropy.at(i) << endl;

        normalizedFeatures.push_back(Features{
                centroids.at(i),
                notNormalizedfeatures.at(i).histogram,
                normCandidatesAverageGreyLevel.at(i),
                normCandidatesContrast.at(i),
                normCandidatesEntropy.at(i),
                normCandidatesSkewness.at(i),
                normCandidatesEnergy.at(i)
        });
    }

    return normalizedFeatures;
}