#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>

using namespace cv;

int PotholeSegmentation(String img_path,
                        const int SuperPixelEdge = 32,
                        const double Cutline_offset = 0.50,
                        const double Density_Threshold = 0.80,
                        const double Variance_Threshold = 0.35,
                        const double Gauss_RoadThreshold = 0.60,
                        const double Rects_X_Offset = 0.0,
                        const double Rects_Y_Offset = 0.9);

#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
