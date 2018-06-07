#ifndef POTHOLEDETENCTIONSYSTEM_UTILIS_H
#define POTHOLEDETENCTIONSYSTEM_UTILIS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "FeaturesExtraction.h"

using namespace cv;
using namespace std;

string set_format(string of_file_name_path,
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

void saveFeatures(const vector<Features> &ft, string directory, string parent, string target);

string extractFileName(const string file_path, const string sep);

#endif //POTHOLEDETENCTIONSYSTEM_UTILIS_H
