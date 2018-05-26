#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include "DataStructures.h"

using namespace cv;
using namespace std;

int PotholeSegmentation(Mat& src,
                        vector<Point> &candidates,
                        const int SuperPixelEdge = 32,
                        const ExtractionThresholds thresholds = defaultThresholds,
                        const Offsets offsets = defaultOffsets);

#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
