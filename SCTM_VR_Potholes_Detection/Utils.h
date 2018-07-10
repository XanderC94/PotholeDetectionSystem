#ifndef POTHOLEDETECTIONSYSTEM_UTILIS_H
#define POTHOLEDETECTIONSYSTEM_UTILIS_H


#include "DataStructures.h"
#include "FeaturesExtraction.h"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

typedef struct Configuration {
    RoadOffsets offsets;
    ExtractionThresholds primaryThresholds;
    ExtractionThresholds secondaryThresholds;
} Configuration;

void portable_mkdir(const char *args);

string set_format(string of_file_name_path,
                string to_new_format,
                bool use_separator = true);

void saveFeaturesJSON(const vector<Features> &features, const string saveDirectory, const vector<string> names,
                      const string saveFile);

void loadFromJSON(const string target, vector<Features> &features, Mat &labels);

Configuration loadProgramConfiguration(const string target);

string extractFileName(const string file_path, const string sep);

vector<String> extractImagePath(const string targets);

void showElaborationStatusToTheUser(string showingWindowTitle, Mat processedImage);

void showElaborationStatusToTheUser(const vector<Features> candidatesFeatures);

void showElaborationStatusToTheUser(const vector<SuperPixel> superPixels);

#endif //POTHOLEDETECTIONSYSTEM_UTILIS_H
