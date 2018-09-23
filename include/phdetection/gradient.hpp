//
// Created by Matteo Gabellini on 15/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_GRADIENTELABORATION_H
#define POTHOLEDETECTIONSYSTEM_GRADIENTELABORATION_H

#include <opencv2/core.hpp>

namespace phd::features {

    typedef struct Gradient {
        cv::Mat x;
        cv::Mat y;
        cv::Mat magnitude;
        cv::Mat direction; //in degrees
    } Gradient;

    void printDirection(Gradient gradient);

    Gradient calculateGradient(cv::Mat candidate);
}

#endif //POTHOLEDETECTIONSYSTEM_GRADIENTELABORATION_H
