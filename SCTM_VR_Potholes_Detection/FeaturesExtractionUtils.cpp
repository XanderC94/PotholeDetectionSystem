//
// Created by Xander_C on 30/06/2018.
//

#include "FeaturesExtractionUtils.h"
#include "HOG.h"

// In order to reduce computation complexity
// calculate the contrast, entropy and energy with in the same loops
void calculateContrastEntropyEnergy(float &outContrast,
                                    float &outEntropy,
                                    float &outEnergy,
                                    const SuperPixel &candidateSuperPixel,
                                    const Mat &candidateGrayScale) {

//    for (int i = 0; i < candidateGrayScale.rows; i++) {
//        for (int j = 0; j < candidateGrayScale.cols; j++) {
//            outContrast = outContrast + powf((i - j), 2) * candidateGrayScale.at<uchar>(i, j);
//            outEntropy = outEntropy + (candidateGrayScale.at<uchar>(i, j) * log10f(candidateGrayScale.at<uchar>(i, j)));
//            outEnergy = outEnergy + powf(candidateGrayScale.at<uchar>(i, j), 2);
//        }
//    }

    for (const Point &coordinates : candidateSuperPixel.points) {
        outContrast += (coordinates.y - coordinates.x) * (coordinates.y - coordinates.x) *
                       candidateGrayScale.at<uchar>(coordinates);
        outEntropy += (candidateGrayScale.at<uchar>(coordinates) * log10f(candidateGrayScale.at<uchar>(coordinates)));
        outEnergy += candidateGrayScale.at<uchar>(coordinates) * candidateGrayScale.at<uchar>(coordinates);
    }

    outEntropy = -outEntropy;
    outEnergy = sqrtf(outEnergy);
}

Mat1f extractHogParams(const Mat &sample, const Mat &candidateGrayScale, const SuperPixel &candidateSuperPixel) {
    HOG hog;
    hog = calculateHOG(sample, defaultConfig);
    int scaleFactor = 5;
    double viz_factor = 5.0;
    vector<OrientedGradientInCell> greaterOrientedGradientsVector = computeGreaterHOGCells(candidateGrayScale,
                                                                                           hog.descriptors,
                                                                                           defaultConfig.cellSize);
//    Mat hogImage = overlapOrientedGradientCellsOnImage(candidateGrayScale,
//                                                       greaterOrientedGradientsVector,
//                                                       defaultConfig.cellSize,
//                                                       scaleFactor,
//                                                       viz_factor);


    auto orientedGradientOfTheSuperPixel = selectNeighbourhoodCellsAtContour(candidateSuperPixel.contour,
                                                                             greaterOrientedGradientsVector);
//    auto orientedGradientOfTheSuperPixel = greaterOrientedGradientsVector;

//    Mat superPixelHogImage = overlapOrientedGradientCellsOnImage(candidateGrayScale,
//                                                                 orientedGradientOfTheSuperPixel,
//                                                                 defaultConfig.cellSize,
//                                                                 scaleFactor,
//                                                                 viz_factor);

    Mat1f hogParams;

    for (auto ogc : orientedGradientOfTheSuperPixel) {
        hogParams.push_back(
                ogc.orientedGradientValue.strength * (ogc.orientedGradientValue.directionInRadians + 1));
    }

    transpose(hogParams, hogParams);

    return hogParams;
}


Mat1b createExclusionMask(const Mat &src, const Mat &sample, const Point2d &tlc,
                          const RoadOffsets &offsets, const ExtractionThresholds &thresholds) {

    Mat1b exclusionMask = Mat::zeros(sample.rows, sample.cols, CV_8UC1);

    for (int i = 0; i < sample.rows; ++i) {
        for (int j = 0; j < sample.cols; ++j) {
            float R = src.at<Vec3b>(tlc + Point2d(j, i)).val[0];
            float G = src.at<Vec3b>(tlc + Point2d(j, i)).val[1];
            float B = src.at<Vec3b>(tlc + Point2d(j, i)).val[2];
            if (isRoad(src.rows, src.cols, offsets, tlc + Point2d(j, i))) {
                exclusionMask.at<uchar>(i, j) = 255;
            }
        }
    }

    return exclusionMask;
}
