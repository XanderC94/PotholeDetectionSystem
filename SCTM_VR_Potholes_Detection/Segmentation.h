#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include "DataStructures.h"

using namespace cv;
using namespace std;

int potholeSegmentation(Mat &src,
                        vector<Point> &candidates,
                        const int superPixelEdge = 32,
                        const ExtractionThresholds thresholds = defaultThresholds,
                        const RoadOffsets offsets = defaultOffsets,
                        const string showingWindowPrefix = "");

int startImageSegmentation(Mat &src,
                           vector<Point> &candidates,
                           const int superPixelEdge = 32,
                           const ExtractionThresholds thresholds = defaultThresholds,
                           const RoadOffsets offsets = defaultOffsets);


#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
