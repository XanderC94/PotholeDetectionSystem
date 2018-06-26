//
// Created by Xander_C on 15/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_HOG_H
#define POTHOLEDETECTIONSYSTEM_HOG_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

typedef struct HOGConfig {
    const int binNumber;
    const Size detenctionWindowSize;
    const Size blockSize;
    const Size blockStride;
    const Size cellSize;
    const Size windowStride;
    const Size padding;
} HOGConfig;

const HOGConfig defaultConfig = {
        9,
        Size(64, 64),   //detenctionWindowSize
        Size(16, 16),     //blockSize
        Size(16, 16),     //blockStride
        Size(2, 2),     //cellSize
        Size(16, 16),     //windowStride
        Size(4, 4)      //padding
};

typedef struct HoG {
    vector<float> descriptors = vector<float>();
    vector<Point> locations = vector<Point>();;
} Hog;

typedef struct GradientLine {
    Point startPoint;
    Point endPoint;
    Scalar color;
    int thickness;
} GradienLine;

typedef struct PointFloat {
    float x;
    float y;
} PointFloat;

typedef struct LineCoordinates {
    PointFloat startPoint;
    PointFloat endPoint;
} LineCoordinates;

typedef struct OrientedGradient {
    float strength;
    float directionInRadians;
} OrientedGradient;

typedef struct OrientedGradientInCell {
    OrientedGradient orientedGradientValue;
    Point cellCenter;
} OrientedGradientInCell;

Mat overlapOrientedGradientCellsOnImage(const Mat &origImg,
                                        vector<OrientedGradientInCell> greaterOrientedGradientCells,
                                        const Size cellSize,
                                        const int scaleFactor,
                                        const double viz_factor);


HoG calculateHoG(const Mat &src, const HOGConfig config = defaultConfig);

vector<OrientedGradientInCell> computeHoGCells(const Mat origImg,
                                               const vector<float> &descriptorValues,
                                               const Size cellSize,
                                               const double viz_factor);

vector<OrientedGradientInCell> computeGreaterHoGCells(const Mat origImg,
                                                      const vector<float> &descriptorValues,
                                                      const Size cellSize,
                                                      const double viz_factor);

vector<float> getHoGDescriptorOnPotholeCorner(vector<float> &descriptorValues);


#endif //POTHOLEDETECTIONSYSTEM_HOG_H