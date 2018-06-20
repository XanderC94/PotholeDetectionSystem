#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include "DataStructures.h"
#include "SuperPixelingUtils.h"

using namespace cv;
using namespace std;

void preprocessing(Mat &src, Mat &processedImage, const double Horizon_Offset);

int extractRegionsOfInterest(const Ptr<SuperpixelLSC> &algorithm,
                             const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                             const int superPixelEdge = 32,
                             const ExtractionThresholds thresholds = defaultThresholds,
                             const RoadOffsets offsets = defaultOffsets);

std::optional<SuperPixel> extractPotholeRegionFromCandidate(const Ptr<SuperpixelLSC> superPixeler,
                                                            const Mat &src,
                                                            const ExtractionThresholds thresholds);

#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
