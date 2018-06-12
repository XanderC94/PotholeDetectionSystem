//
// Created by Matteo Gabellini on 25/05/2018.
//
typedef struct RoadOffsets {
    double Horizon_Offset;
    double SLine_X_Offset;
    double SLine_Y_Offset;
} Offsets;
const RoadOffsets defaultOffsets = {0.60, 0.0, 0.8};


void printOffsets(RoadOffsets of);


typedef struct ExtractionThresholds {
    double Density_Threshold;
    double Variance_Threshold;
    double Gauss_RoadThreshold;
} ExtractionThresholds;

const ExtractionThresholds defaultThresholds = {0.80, 0.35, 0.60};


void printThresholds(ExtractionThresholds thresholds);