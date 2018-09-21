//
// Created by Matteo Gabellini on 25/05/2018.
//
#ifndef POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H
#define POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H

#include <opencv2/core.hpp>
#include <set>

using namespace cv;
using namespace std;

namespace phd::ontologies {

    typedef struct RoadOffsets {
        double horizon;
        double xRightOffset;
        double xLeftOffset;
        double yRightOffset;
        double yLeftOffset;
        double rightEscapeOffset;
        double leftEscapeOffset;
    } RoadOffsets;
    const RoadOffsets defaultOffsets = {0.60, 0.0, 0.4, 0.8, 0.8, 0.5, 0.5};

    typedef struct ExtractionThresholds {
        double density;
        double variance;
        double gauss;
        double minGreyRatio;
        double maxGreyRatio;
        double minGreenRatio;
    } ExtractionThresholds;

    const ExtractionThresholds defaultThresholds = {0.80, 0.35, 0.60, 1.25, 4.0, 1.15};

    typedef struct Features {
        int _class;
        int label;
        int id;
        Mat candidate;
        Mat histogram;
        float averageGreyValue;
        Scalar averageRGBValues;
        float contrast;
        float entropy;
        float skewness;
        float energy;
        Mat1f hogDescriptors;
    } Features;

    typedef struct SuperPixel {
        int label;
        vector<cv::Point> points;
        Point2d center;
        Mat selection;
        Mat1b mask;
        Mat contour;
        Scalar meanColour;
        std::set<int> neighbors;
    } SuperPixel;

    typedef struct FeaturesVectors {
        vector<float> averageGreyLevels = vector<float>();
        vector<float> averageRedLevels = vector<float>();
        vector<float> averageGreenLevels = vector<float>();
        vector<float> averageBlueLevels = vector<float>();
        vector<Mat> histograms = vector<Mat>();
        vector<float> contrasts = vector<float>();
        vector<float> entropies = vector<float>();
        vector<float> skewnesses = vector<float>();
        vector<float> energies = vector<float>();
        vector<Mat1f> hogDescriptors = vector<Mat1f>();
    } FeaturesVectors;

    enum ClassificationClasses {
        streetSideWalkOrCar = -1,
        outOfRoad = -2,
        asphaltCrack = 2,
        pothole = 1
    };

    void printThresholds(ExtractionThresholds thresholds);


    void printOffsets(RoadOffsets of);
}

#endif //POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H