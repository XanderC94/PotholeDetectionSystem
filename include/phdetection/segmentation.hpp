#ifndef POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETECTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include "ontologies.hpp"
#include "superpixeling.hpp"

namespace phd{
    namespace segmentation {

        void preprocessing(cv::Mat &src, cv::Mat &processedImage, const double Horizon_Offset);

        int extractRegionsOfInterest(const cv::Ptr<cv::ximgproc::SuperpixelLSC> &algorithm,
                const cv::Mat &src, std::vector<phd::ontologies::SuperPixel> &candidateSuperpixels,
                const int superPixelEdge = 32,
                const phd::ontologies::ExtractionThresholds thresholds = phd::ontologies::defaultThresholds,
                const phd::ontologies::RoadOffsets offsets = phd::ontologies::defaultOffsets
                        );

        std::vector<phd::ontologies::SuperPixel>
        extractPotholeRegionFromCandidate (
                const cv::Mat &candidate,
                const cv::Mat1b &exclusionMask,
                const phd::ontologies::ExtractionThresholds &thresholds
                );
    }
}

#endif //POTHOLEDETECTIONSYSTEM_SEGMENTATION_H
