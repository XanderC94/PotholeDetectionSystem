//
// Created by Xander_C on 27/06/2018.
//

#ifndef POTHOLEDETECTIONSYSTEM_MLUTILS_H
#define POTHOLEDETECTIONSYSTEM_MLUTILS_H

#include <opencv2/core.hpp>
#include "DataStructures.h"

namespace mlutils {
    Mat ConvertFeatures(const vector<Features> &features);

    Mat ConvertHOGFeatures(const vector<Features> &features, const int var_count);

}

#endif //POTHOLEDETECTIONSYSTEM_MLUTILS_H
