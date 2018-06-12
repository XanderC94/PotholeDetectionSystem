//
// Created by Matteo Gabellini on 11/06/2018.
//
#include "DataStructures.h"
#include <iostream>

using namespace std;

void printOffsets(RoadOffsets of) {
    cout << "ROAD OFFSETS" << endl;
    cout << "Horizon_Offset: " << of.Horizon_Offset << endl;
    cout << "SLine_X_Offset: " << of.SLine_X_Offset << endl;
    cout << "SLine_Y_Offset: " << of.SLine_Y_Offset << endl;
}

void printThresholds(ExtractionThresholds thresholds) {
    cout << "THRESHOLDS" << endl;
    cout << "Density_Threshold: " << thresholds.Density_Threshold << endl;
    cout << "Variance_Threshold: " << thresholds.Variance_Threshold << endl;
    cout << "Gauss_RoadThreshold: " << thresholds.Gauss_RoadThreshold << endl;
}
