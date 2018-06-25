#ifndef POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETECTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include "DataStructures.h"
#include "SuperPixelingUtils.h"
#include "Optional.h"

using namespace cv;
using namespace std;

void preprocessing(Mat &src, Mat &processedImage, const double Horizon_Offset);

int extractRegionsOfInterest(const Ptr<SuperpixelLSC> &algorithm,
                             const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                             const int superPixelEdge = 32,
                             const ExtractionThresholds thresholds = defaultThresholds,
                             const RoadOffsets offsets = defaultOffsets);

cv::Optional<SuperPixel> extractPotholeRegionFromCandidate(const Mat &candidate, const Mat1b &roadMask,
                                                           const ExtractionThresholds &thresholds);

#endif //POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
