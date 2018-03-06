#ifndef POTHOLEDETENCTIONSYSTEM_HISTOGRAMELABORATION_H
#define POTHOLEDETENCTIONSYSTEM_HISTOGRAMELABORATION_H

#include <cstdio>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <iterator>

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace cv::ximgproc;


Mat ExtractHistograms(Mat src);

#endif //POTHOLEDETENCTIONSYSTEM_HISTOGRAMELABORATION_H
