#ifndef POTHOLEDETENCTIONSYSTEM_UTILIS_H
#define POTHOLEDETENCTIONSYSTEM_UTILIS_H

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


void set_format(string &of_file_name_path,
                string to_new_format,
                bool use_separator = true);

void load_from_directory(const string &directory,
                         vector<string> &ids,
                         vector<Mat> &set,
                         Mat &labels,
                         int label = -1,
                         int image_type = IMREAD_COLOR);

int resize_all_in(const string parent,
                  const string folder,
                  const int width = 1280,
                  const int height = 720);

#endif //POTHOLEDETENCTIONSYSTEM_UTILIS_H
