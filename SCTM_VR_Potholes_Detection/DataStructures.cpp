//
// Created by Matteo Gabellini on 11/06/2018.
//
#include "DataStructures.h"
#include <iostream>

using namespace std;

void printOffsets(RoadOffsets of) {
    cout << "ROAD OFFSETS" << endl
         << "Horizon_Offset: " << of.Horizon_Offset << endl
         << "SLine_X_Right_Offset: " << of.SLine_X_Right_Offset << endl
         << "SLine_X_Left_Offset: " << of.SLine_X_Left_Offset << endl
         << "SLine_Y_Offset: " << of.SLine_Y_Right_Offset << endl
         << "SLine_Y_Offset: " << of.SLine_Y_Left_Offset << endl;
}

void printThresholds(ExtractionThresholds thresholds) {
    cout << "THRESHOLDS" << endl
         << "Density_Threshold: " << thresholds.Density_Threshold << endl
         << "Variance_Threshold: " << thresholds.Variance_Threshold << endl
         << "Gauss_RoadThreshold: " << thresholds.Gauss_RoadThreshold << endl
         << "colourRatioThresholdMin: " << thresholds.colourRatioThresholdMin << endl
         << "colourRatioThresholdMax: " << thresholds.colourRatioThresholdMax << endl;
}
