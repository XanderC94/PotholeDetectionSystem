#ifndef POTHOLEDETECTIONSYSTEM_UTILIS_H
#define POTHOLEDETECTIONSYSTEM_UTILIS_H


#include "ontologies.hpp"
#include <opencv2/core.hpp>

//using namespace cv;
//using namespace std;
//
//using namespace phd::ontologies;

namespace phd::io {

    typedef struct Configuration {
        phd::ontologies::RoadOffsets offsets;
        phd::ontologies::ExtractionThresholds primaryThresholds;
        phd::ontologies::ExtractionThresholds secondaryThresholds;
    } Configuration;

    std::string GetCurrentWorkingDir(void);

    void portable_mkdir(const char *args);

    std::string set_format(std::string of_file_name_path,
                           std::string to_new_format,
                      bool use_separator = true);

    void saveFeaturesJSON(const std::vector<phd::ontologies::Features> &features, const std::string saveDirectory, const std::vector<std::string> names,
                          const std::string saveFile);

    void loadFromJSON(const std::string target, std::vector<phd::ontologies::Features> &features, cv::Mat &labels);

    Configuration loadProgramConfiguration(const std::string target);

    std::string extractFileName(const std::string file_path, const std::string sep);

    std::vector<std::string> extractImagePath(const std::string targets);

    void showElaborationStatusToTheUser(std::string showingWindowTitle, cv::Mat processedImage);

    void showElaborationStatusToTheUser(const std::vector<phd::ontologies::Features> candidatesFeatures);

    void showElaborationStatusToTheUser(const std::vector<phd::ontologies::SuperPixel> superPixels);
}

#endif //POTHOLEDETECTIONSYSTEM_UTILIS_H
