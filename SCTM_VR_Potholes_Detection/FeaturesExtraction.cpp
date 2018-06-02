//
// Created by Matteo Gabellini on 02/06/2018.
//

#include "FeaturesExtraction.h"

vector<Features> extractFeatures(Mat sourceImage, vector<Point> centroids, Size candidate_size) {

    auto candidates = vector<Mat>();
    auto features = vector<Features>();

    auto candidatesAverageGreyLevel = vector<double>();
    auto candidatesContrast = vector<double>();
    auto candidatesEntropy = vector<double>();
    auto candidatesSkewness = vector<double>();


    Mat candidateGrayScale;

    /*
        * Each centroid of superpixels extracted in the previous phase
        *  1. Candidate will be converted to greyscale
        *  2. The histogram will be calculated
        *  3. Calculate the average gray value
        *  4. Calculate the contrast
        *  5. Calculate 3-order moments (is Skewness according to http://aishack.in/tutorials/image-moments/)
        *  6. Calculate Consistency
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

        double contrast = 0.0;
        double entropy = 0.0;

        for (int i = 0; i < candidateGrayScale.rows; i++) {
            for (int j = 0; j < candidateGrayScale.cols; j++) {
                contrast = contrast + pow((i - j), 2) * candidateGrayScale.at<uchar>(i, j);
                entropy =
                        entropy + (candidateGrayScale.at<uchar>(i, j) * log10(candidateGrayScale.at<uchar>(i, j)));
            }
        }

        candidatesContrast.push_back(contrast);

        entropy = 0 - entropy;

        //candidatesEntropy.push_back(entropy);

        Scalar averageGreyValue = mean(candidateGrayScale);
        candidatesAverageGreyLevel.push_back(averageGreyValue[0]);
        //cout << "Valore Medio " << averageGreyValue[0] << endl;

        double skewness = calculateSkewnessGrayImage(candidate, averageGreyValue[0]);
        candidatesSkewness.push_back(skewness);
        //cout << "Skewness " << skewness << endl;

        auto c_name = "Candidate @ (" + to_string(c.x) + ", " + to_string(c.y) + ")";
        imshow(c_name, candidateGrayScale);

        cout << c_name << endl;
        cout <<
             "Average Gray Value: " << averageGreyValue[0] <<
             " Contrast: " << contrast <<
             " Skeweness: " << skewness <<
             " Entropy: " << entropy << endl;

        Mat histogram = ExtractHistograms(candidateGrayScale);

        features.push_back(Features{c, histogram, averageGreyValue, contrast, entropy, skewness});
    }

    return features;
}