//
// Created by Matteo Gabellini on 25/05/2018.
//
#ifndef POTHOLEDETENCTIONSYSTEM_DATASTRUCTURES_H
#define POTHOLEDETENCTIONSYSTEM_DATASTRUCTURES_H

#include <opencv2/core.hpp>
using namespace cv;

typedef struct RoadOffsets {
    double Horizon_Offset;
    double SLine_X_Offset;
    double SLine_Y_Offset;
} RoadOffsets;
const RoadOffsets defaultOffsets = {0.60, 0.0, 0.8};

void printOffsets(RoadOffsets of);

typedef struct ExtractionThresholds {
    double Density_Threshold;
    double Variance_Threshold;
    double Gauss_RoadThreshold;
} ExtractionThresholds;

const ExtractionThresholds defaultThresholds = {0.80, 0.35, 0.60};

typedef struct Features {
    Mat candidate;
    Mat histogram;
    float averageGreyValue;
    float contrast;
    float entropy;
    float skewness;
    float energy;
} Features;

void printThresholds(ExtractionThresholds thresholds);

#endif //POTHOLEDETENCTIONSYSTEM_DATASTRUCTURES_H