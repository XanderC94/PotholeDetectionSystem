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
            assert(ft.HasMember("id"));
            assert(ft.HasMember("contrast"));
            assert(ft.HasMember("avgGreyValue"));
            assert(ft.HasMember("avgRGBValues"));
            assert(ft.HasMember("skewness"));
            assert(ft.HasMember("energy"));
            assert(ft.HasMember("entropy"));
            assert(ft.HasMember("hog"));

            assert(ft["class"].IsInt());
            assert(ft["label"].IsInt());
            assert(ft["contrast"].IsFloat());
            assert(ft["avgGreyValue"].IsFloat());
            assert(ft["avgRGBValues"].IsArray());
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
                    ft["id"].GetInt(),
                    Mat(),
                    Mat(),
                    ft["avgGreyValue"].GetFloat(),
                    Scalar(
                            ft["avgRGBValues"].GetArray()[0].GetDouble(),
                            ft["avgRGBValues"].GetArray()[1].GetDouble(),
                            ft["avgRGBValues"].GetArray()[2].GetDouble()
                    ),
                    ft["contrast"].GetFloat(),
                    ft["energy"].GetFloat(),
                    ft["skewness"].GetFloat(),
                    ft["entropy"].GetFloat(),
                    hog
            };

            features.push_back(f);
            labels.push_back(ft["class"].GetInt());
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
        sw.Key("id");
        sw.Int(ft.id);
        sw.Key("contrast");
        sw.Double(ft.contrast);
        sw.Key("avgGreyValue");
        sw.Double(ft.averageGreyValue);
        sw.Key("avgRGBValues");
        sw.StartArray();
        for (const double channel : ft.averageRGBValues.val) sw.Double(channel);
        sw.EndArray();
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

        imwrite("../" + saveDirectory + "/" + c_name + "_L" + to_string(ft.label) + "_" + to_string(ft.id) + ".bmp",
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

Configuration loadProgramConfiguration(const string target) {

    RoadOffsets offsets;
    ExtractionThresholds primary;
    ExtractionThresholds secondary;

    ifstream json(target, fstream::in);

    IStreamWrapper wrapper(json);

    Document config;

    config.ParseStream(wrapper);

    if (json.is_open() && config.IsObject()) {

        cout << "Opened file " << target << endl;

        assert(config.HasMember("offsets"));
        assert(config["offsets"].IsObject());

        assert(config["offsets"].HasMember("horizon"));
        assert(config["offsets"]["horizon"].IsDouble());

        assert(config["offsets"].HasMember("xRightOffset"));
        assert(config["offsets"]["xRightOffset"].IsDouble());
        assert(config["offsets"].HasMember("yRightOffset"));
        assert(config["offsets"]["yRightOffset"].IsDouble());
        assert(config["offsets"].HasMember("rightEscapeOffset"));
        assert(config["offsets"]["rightEscapeOffset"].IsDouble());

        assert(config["offsets"].HasMember("xLeftOffset"));
        assert(config["offsets"]["xLeftOffset"].IsDouble());
        assert(config["offsets"].HasMember("yLeftOffset"));
        assert(config["offsets"]["yLeftOffset"].IsDouble());
        assert(config["offsets"].HasMember("leftEscapeOffset"));
        assert(config["offsets"]["leftEscapeOffset"].IsDouble());

        offsets.horizon = config["offsets"]["horizon"].GetDouble();
        offsets.leftEscapeOffset = config["offsets"]["leftEscapeOffset"].GetDouble();
        offsets.rightEscapeOffset = config["offsets"]["rightEscapeOffset"].GetDouble();
        offsets.xLeftOffset = config["offsets"]["xLeftOffset"].GetDouble();
        offsets.xRightOffset = config["offsets"]["xRightOffset"].GetDouble();
        offsets.yLeftOffset = config["offsets"]["yLeftOffset"].GetDouble();
        offsets.yRightOffset = config["offsets"]["yRightOffset"].GetDouble();

        assert(config.HasMember("thresholds"));
        assert(config["thresholds"].IsObject());

        assert(config["thresholds"].HasMember("primary"));
        assert(config["thresholds"]["primary"].IsObject());

        assert(config["thresholds"]["primary"].HasMember("density"));
        assert(config["thresholds"]["primary"]["density"].IsObject());
        assert(config["thresholds"]["primary"].HasMember("variance"));
        assert(config["thresholds"]["primary"]["variance"].IsObject());
        assert(config["thresholds"]["primary"].HasMember("gauss"));
        assert(config["thresholds"]["primary"]["gauss"].IsObject());
        assert(config["thresholds"]["primary"].HasMember("minGreyRatio"));
        assert(config["thresholds"]["primary"]["minGreyRatio"].IsObject());
        assert(config["thresholds"]["primary"].HasMember("maxGreyRatio"));
        assert(config["thresholds"]["primary"]["maxGreyRatio"].IsObject());
        assert(config["thresholds"]["primary"].HasMember("minGreenRatio"));
        assert(config["thresholds"]["primary"]["minGreenRatio"].IsObject());

        primary.density = config["thresholds"]["primary"]["density"].GetDouble();
        primary.variance = config["thresholds"]["primary"]["variance"].GetDouble();
        primary.gauss = config["thresholds"]["primary"]["gauss"].GetDouble();
        primary.minGreyRatio = config["thresholds"]["primary"]["minGreyRatio"].GetDouble();
        primary.maxGreyRatio = config["thresholds"]["primary"]["maxGreyRatio"].GetDouble();
        primary.minGreenRatio = config["thresholds"]["primary"]["minGreenRatio"].GetDouble();

        assert(config["thresholds"].HasMember("secondary"));
        assert(config["thresholds"]["secondary"].IsObject());

        assert(config["thresholds"]["secondary"].HasMember("density"));
        assert(config["thresholds"]["secondary"]["density"].IsObject());
        assert(config["thresholds"]["secondary"].HasMember("variance"));
        assert(config["thresholds"]["secondary"]["variance"].IsObject());
        assert(config["thresholds"]["secondary"].HasMember("gauss"));
        assert(config["thresholds"]["secondary"]["gauss"].IsObject());
        assert(config["thresholds"]["secondary"].HasMember("minGreyRatio"));
        assert(config["thresholds"]["secondary"]["minGreyRatio"].IsObject());
        assert(config["thresholds"]["secondary"].HasMember("maxGreyRatio"));
        assert(config["thresholds"]["secondary"]["maxGreyRatio"].IsObject());
        assert(config["thresholds"]["secondary"].HasMember("minGreenRatio"));
        assert(config["thresholds"]["secondary"]["minGreenRatio"].IsObject());

        secondary.density = config["thresholds"]["secondary"]["density"].GetDouble();
        secondary.variance = config["thresholds"]["secondary"]["variance"].GetDouble();
        secondary.gauss = config["thresholds"]["secondary"]["gauss"].GetDouble();
        secondary.minGreyRatio = config["thresholds"]["secondary"]["minGreyRatio"].GetDouble();
        secondary.maxGreyRatio = config["thresholds"]["secondary"]["maxGreyRatio"].GetDouble();
        secondary.minGreenRatio = config["thresholds"]["secondary"]["minGreenRatio"].GetDouble();

    } else {
        cerr << "Program configuration is missing... Check it's existence or create a new config.json"
             << " under the ../config/ folder inside the program directory." << target << endl;

        exit(-3);
    }

    return Configuration{offsets, primary, secondary};
}
