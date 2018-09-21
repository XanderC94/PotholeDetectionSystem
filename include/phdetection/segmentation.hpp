#ifndef POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETECTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include "ontologies.hpp"
#include "superpixeling.hpp"
#include "optional.hpp"

using namespace cv;
using namespace cv::ximgproc;
using namespace phd::ontologies;

namespace phd::segmentation {

    void preprocessing(Mat &src, Mat &processedImage, const double Horizon_Offset);

    int extractRegionsOfInterest(const Ptr<SuperpixelLSC> &algorithm,
                                 const Mat &src, vector<SuperPixel> &candidateSuperpixels,
                                 const int superPixelEdge = 32,
                                 const ExtractionThresholds thresholds = defaultThresholds,
                                 const RoadOffsets offsets = defaultOffsets);

    std::vector<SuperPixel> extractPotholeRegionFromCandidate(const Mat &candidate, const Mat1b &exclusionMask,
                                                              const ExtractionThresholds &thresholds);
}

#endif //POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
