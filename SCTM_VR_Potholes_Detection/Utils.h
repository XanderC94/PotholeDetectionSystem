#ifndef POTHOLEDETENCTIONSYSTEM_UTILIS_H
#define POTHOLEDETENCTIONSYSTEM_UTILIS_H


#include "DataStructures.h"
#include "FeaturesExtraction.h"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

void portable_mkdir(const char *args);

string set_format(string of_file_name_path,
                string to_new_format,
                bool use_separator = true);

void loadFromCSV(const string target,
                 vector<Features> &ft,
                 Mat &labels);

int resize_all_in(const string parent,
                  const string folder,
                  const int width = 1280,
                  const int height = 720);

void saveFeatures(const vector<Features> &features, const string saveDirectory, const vector<string> names,
                  const string saveFile);

string extractFileName(const string file_path, const string sep);

vector<String> extractImagePath(const string targets);

#endif //POTHOLEDETENCTIONSYSTEM_UTILIS_H
