//
// Created by Matteo Gabellini on 11/06/2018.
//
#include "../include/phdetection/ontologies.hpp"
#include <iostream>

using namespace std;

namespace phd {
    namespace ontologies {
        void printOffsets(RoadOffsets of) {
            cout << "ROAD OFFSETS:" << endl
            << "\thorizon: " << of.horizon << endl

            << "\txRightOffset: " << of.xRightOffset << endl
            << "\tyRightOffset: " << of.yRightOffset << endl
            << "\trightEscapeOffset: " << of.rightEscapeOffset << endl

            << "\txLeftOffset: " << of.xLeftOffset << endl
            << "\tyLeftOffset: " << of.yLeftOffset << endl
            << "\tleftEscapeOffset: " << of.leftEscapeOffset << endl
            << endl;
        }

        void printThresholds(ExtractionThresholds thresholds) {
            cout << "THRESHOLDS" << endl
            << "\tdensity: " << thresholds.density << endl
            << "\tvariance: " << thresholds.variance << endl
            << "\tgauss: " << thresholds.gauss << endl
            << "\tminGreyRatio: " << thresholds.minGreyRatio << endl
            << "\tmaxGreyRatio: " << thresholds.maxGreyRatio << endl
            << "\tminGreenRatio: " << thresholds.minGreenRatio << endl
            << endl;
        }

    }
}