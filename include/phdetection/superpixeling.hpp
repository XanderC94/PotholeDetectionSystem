//
// Created by Matteo Gabellini on 11/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H
#define POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H


#include <opencv2/core.hpp>
#include <opencv2/ximgproc.hpp>
#include "ontologies.hpp"

namespace phd {
    namespace superpixeling {

        std::vector<std::vector<cv::Point>> getContours(const cv::Mat &mask);

        cv::Mat getContoursMask(const cv::Mat &mask);

        phd::ontologies::SuperPixel getSuperPixel(const cv::Mat &src, int superPixelLabel,
                const cv::Mat &labels);

        phd::ontologies::SuperPixel getSuperPixel(const cv::Mat &src, int superPixelLabel,
                const cv::Mat &labels, const phd::ontologies::RoadOffsets offsets);

        phd::ontologies::SuperPixel getSuperPixel(const cv::Mat &src, const cv::Mat1b &roadMask,
                const int superPixelLabel, const cv::Mat &labels);

        bool isRoad(const int H, const int W, const phd::ontologies::RoadOffsets offsets, const cv::Point2d center);

        cv::Ptr<cv::ximgproc::SuperpixelLSC> initSuperPixelingLSC(const cv::Mat &src, const int superPixelEdge);

        cv::Ptr<cv::ximgproc::SuperpixelSLIC> initSuperPixelingSLIC(const cv::Mat &src, const int superPixelEdge, const float ruler);

        cv::Point2d calculateSuperPixelCenter(std::vector<cv::Point> pixelOfTheSuperPixel);

        cv::Point2d calculateSuperPixelVariance(std::vector<cv::Point> superPixel, cv::Point2d center);

        double calculateSuperPixelDensity(std::vector<cv::Point> superPixel);
    }
}

#endif //POTHOLEDETECTIONSYSTEM_SUPERPIXELINGUTILS_H
