#ifndef POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
#define POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

int PotholeSegmentation(Mat& src,
						vector<Point> &candidates,
                        const int SuperPixelEdge = 32,
						const double Cutline_offset = 0.60,
						const double Density_Threshold = 0.80,
						const double Variance_Threshold = 0.35,
						const double Gauss_RoadThreshold = 0.60,
						const double Rects_X_Offset = 0.0,
						const double Rects_Y_Offset = 0.8);

#endif //POTHOLEDETENCTIONSYSTEM_SEGMENTATION_H
