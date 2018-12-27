//
// Created by Matteo Gabellini on 25/05/2018.
//
#ifndef POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H
#define POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H

#include <opencv2/core.hpp>
#include <set>

namespace phd {
    namespace ontologies {

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
            cv::Mat candidate;
            cv::Mat histogram;
            float averageGreyValue;
            cv::Scalar averageRGBValues;
            float contrast;
            float entropy;
            float skewness;
            float energy;
            cv::Mat1f hogDescriptors;
        } Features;

        typedef struct SuperPixel {
            int label;
            std::vector<cv::Point> points;
            cv::Point2d center;
            cv::Mat selection;
            cv::Mat1b mask;
            cv::Mat contour;
            cv::Scalar meanColour;
            std::set<int> neighbors;
        } SuperPixel;

        typedef struct FeaturesVectors {
            std::vector<float> averageGreyLevels = std::vector<float>();
            std::vector<float> averageRedLevels = std::vector<float>();
            std::vector<float> averageGreenLevels = std::vector<float>();
            std::vector<float> averageBlueLevels = std::vector<float>();
            std::vector<cv::Mat> histograms = std::vector<cv::Mat>();
            std::vector<float> contrasts = std::vector<float>();
            std::vector<float> entropies = std::vector<float>();
            std::vector<float> skewnesses = std::vector<float>();
            std::vector<float> energies = std::vector<float>();
            std::vector<cv::Mat1f> hogDescriptors = std::vector<cv::Mat1f>();
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
}

#endif //POTHOLEDETECTIONSYSTEM_DATASTRUCTURES_H