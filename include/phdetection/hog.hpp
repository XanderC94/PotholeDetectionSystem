//
// Created by Xander_C on 15/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_HOG_H
#define POTHOLEDETECTIONSYSTEM_HOG_H

#include <opencv2/core.hpp>

namespace phd::features {

    typedef struct HOGConfig {
        const int binNumber;
        const cv::Size detectionWindowSize;
        const cv::Size blockSize;
        const cv::Size blockStride;
        const cv::Size cellSize;
        const cv::Size windowStride;
        const cv::Size padding;
    } HOGConfig;

    const HOGConfig defaultConfig = {
            9,
            cv::Size(64, 64),   //detectionWindowSize
            cv::Size(16, 16),     //blockSize
            cv::Size(16, 16),     //blockStride
            cv::Size(2, 2),     //cellSize
            cv::Size(16, 16),     //windowStride
            cv::Size(4, 4)      //padding
    };

    typedef struct HOG {
        std::vector<float> descriptors = std::vector<float>();
        std::vector<cv::Point> locations = std::vector<cv::Point>();;
    } Hog;

    typedef struct GradientLine {
        cv::Point startPoint;
        cv::Point endPoint;
        cv::Scalar color;
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
        cv::Point cellCenter;
    } OrientedGradientInCell;

    cv::Mat overlapOrientedGradientCellsOnImage(
            const cv::Mat &origImg,
            std::vector<OrientedGradientInCell> greaterOrientedGradientCells,
            const cv::Size cellSize,
            const int scaleFactor,
            const double viz_factor
        );


    HOG calculateHOG(const cv::Mat &src, const HOGConfig config = defaultConfig);

    std::vector<OrientedGradientInCell> computeHOGCells(
            const cv::Mat origImg,
            const std::vector<float> &descriptorValues,
            const cv::Size cellSize
        );

    std::vector<OrientedGradientInCell> computeGreaterHOGCells(
            const cv::Mat origImg,
            const std::vector<float> &descriptorValues,
            const cv::Size cellSize
        );

    std::vector<float> getHOGDescriptorOnPotholeCorner(std::vector<float> &descriptorValues);

    std::vector<OrientedGradientInCell> selectNeighbourhoodCellsAtContour(
            cv::Mat contoursMask,
            std::vector<OrientedGradientInCell> orientedGradientsCells,
            int neighbourhood = 2
        );
}

#endif //POTHOLEDETECTIONSYSTEM_HOG_H