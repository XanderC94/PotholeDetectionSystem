#include "Utils.h"
#include <sys/stat.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>


#include <unistd.h>
#define getCurrentDir getcwd

using namespace rapidjson;

std::string getCurrentWorkingDir(void) {
    char buff[FILENAME_MAX];
    getCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);
    return current_working_dir;
}

int resize_all_in(const string parent, const string folder, const int width, const int height) {

    vector<String> fn;
    glob(folder + "/*", fn);

    cout << "Found " << fn.size() << " images..." << endl;

    for (int i = 0; i < fn.size(); i++) {
        string file_name = fn[i];
        Mat img = imread(file_name, IMREAD_COLOR);

        if (img.empty()) {
            cerr << "invalid image: " << file_name << endl;
            continue;
        } else {
            cout << "Loaded image " << file_name << endl;

            Mat dst;
            resize(img, dst, Size(width, height));

            imwrite(parent + "/scaled/" + file_name.substr(file_name.find_last_of("\\")), dst);
        }
    }

    return 1;
}

void portable_mkdir(const char *args) {

#if defined(_WIN32) || defined(_WIN32_WINNT) || defined(_WIN64)
    mkdir(args);
#else
    mkdir(args, S_IWUSR);
#endif
}

string extractFileName(string file_path, const string sep = "/") {

    std::replace(file_path.begin(), file_path.end(), '\\', '/');

    auto offset = file_path.find_last_of(sep);

    return file_path.substr(offset + 1);
}

string set_format(string of_file_name_path, string to_new_format, bool use_separator) {
    int image_format_offset_begin = of_file_name_path.find_last_of(".");
    string image_format = of_file_name_path.substr(image_format_offset_begin);
    int image_format_offset_end = image_format.length() + image_format_offset_begin;

    return of_file_name_path.replace(image_format_offset_begin, image_format_offset_end,
                              (use_separator ? "." : "") + to_new_format);
}

void loadFromJSON(const string target, vector<Features> &features, Mat &labels) {

    ifstream json(target, fstream::in);

    IStreamWrapper wrapper(json);

    Document document;

    document.ParseStream(wrapper);

    if (json.is_open() && document.IsObject()) {

        cout << "Opened file " << target << endl;

        for (const auto &ft : document["features"].GetArray()) {

            assert(ft.HasMember("class"));
            assert(ft.HasMember("label"));
            assert(ft.HasMember("contrast"));
            assert(ft.HasMember("avgGreyValue"));
            assert(ft.HasMember("skewness"));
            assert(ft.HasMember("energy"));
            assert(ft.HasMember("entropy"));
            assert(ft.HasMember("hog"));

            assert(ft["class"].IsInt());
            assert(ft["label"].IsInt());
            assert(ft["contrast"].IsFloat());
            assert(ft["avgGreyValue"].IsFloat());
            assert(ft["skewness"].IsFloat());
            assert(ft["energy"].IsFloat());
            assert(ft["entropy"].IsFloat());
            assert(ft["hog"].IsArray());

            Mat1f hog;

            for (const auto &value : ft["hog"].GetArray()) {
                hog.push_back(value.GetFloat());
            }

            transpose(hog, hog);

            Features f = {
                    ft["class"].GetInt(),
                    ft["label"].GetInt(),
                    0,
                    Mat(),
                    Mat(),
                    ft["avgGreyValue"].GetFloat(),
                    ft["contrast"].GetFloat(),
                    ft["energy"].GetFloat(),
                    ft["skewness"].GetFloat(),
                    ft["entropy"].GetFloat(),
                    hog
            };

            features.push_back(f);
            labels.push_back(ft["label"].GetInt());
        }

    } else {
        cerr << "Unable to open file or the json file is malformed" << target << endl;
    }

    cout << labels.rows << " " << features.size() << endl;

}

void saveFeaturesJSON(const vector<Features> &features, const string saveDirectory, const vector<string> names,
                      const string saveFile) {

    portable_mkdir(("../" + saveDirectory).data());

    ofstream json("../" + saveDirectory + "/" + saveFile + ".json", fstream::out | fstream::app);

    OStreamWrapper wrapper(json);
    PrettyWriter<OStreamWrapper> sw(wrapper);

    sw.StartObject();
    sw.Key("features");
    sw.StartArray();

    for (int i = 0; i < features.size(); ++i) {

        const auto ft = features[i];

        const auto c_name = set_format(extractFileName(names[i]), "", false);

        sw.StartObject();
        sw.Key("class");
        sw.Int(ft._class);
        sw.Key("sample");
        sw.String(c_name.data());
        sw.Key("label");
        sw.Int(ft.label);
        sw.Key("contrast");
        sw.Double(ft.contrast);
        sw.Key("avgGreyValue");
        sw.Double(ft.averageGreyValue);
        sw.Key("skewness");
        sw.Double(ft.skewness);
        sw.Key("energy");
        sw.Double(ft.energy);
        sw.Key("entropy");
        sw.Double(ft.entropy);
        sw.Key("hog");
        sw.StartArray();

        for (unsigned i = 0; i < ft.hogDescriptors.cols; i++) sw.Double(ft.hogDescriptors.at<float>(0, i));

        sw.EndArray();
        sw.EndObject();

        sw.Flush();

        imwrite("../" + saveDirectory + "/" + c_name + "_L" + to_string(ft.label) + "_" + to_string(ft.uid) + ".bmp",
                ft.candidate);
    }

    sw.EndArray();
    sw.EndObject();

    json.close();

}

vector<String> extractImagePath(const string targets) {

    vector<String> res;
    vector<String> fnJpg;
    vector<String> fnPng;
    vector<String> fnBmp;

    glob(targets + "/*.jpg", fnJpg);
    glob(targets + "/*.png", fnPng);
    glob(targets + "/*.bmp", fnBmp);

    for (auto jpgImage : fnJpg) {
        res.push_back(jpgImage);
    }
    for (auto pngImage : fnPng) {
        res.push_back(pngImage);
    }
    for (auto bmpImage : fnBmp) {
        res.push_back(bmpImage);
    }

    return res;
}