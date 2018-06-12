#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include "DataStructures.h"
#include "SuperPixelingUtils.h"

using namespace cv;
using namespace std;

int extractPossiblePotholes(Mat &src,
                            vector<Point> &candidateCentroids,
                            const int superPixelEdge = 32,
                            const ExtractionThresholds thresholds = defaultThresholds,
                            const RoadOffsets offsets = defaultOffsets,
                            const string showingWindowPrefix = "");

int initialImageSegmentation(Mat &src,
                             vector<Point> &candidateCentroids,
                             const int superPixelEdge = 32,
                             const ExtractionThresholds thresholds = defaultThresholds,
                             const RoadOffsets offsets = defaultOffsets);

SuperPixel extractPotholeRegionFromCandidate(Mat &src, string candidateName);

#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
