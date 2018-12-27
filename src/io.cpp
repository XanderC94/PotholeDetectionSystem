#include <phdetection/io.hpp>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <libgen.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>

#if defined(_WIN32)
#define WINDOWS
#elif defined(_WIN64)
#define WINDOWS
#endif

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

using namespace rapidjson;
using namespace cv;
using namespace phd::ontologies;
using namespace std;

namespace phd::io {

    std::string GetCurrentWorkingDir( void ) {
        char buff[FILENAME_MAX];
        GetCurrentDir( buff, FILENAME_MAX );
        std::string current_working_dir(buff);
        return current_working_dir;
    }

    void portable_mkdir(const char *args) {

#ifdef WINDOWS
        mkdir(args);
#else
        mkdir(args, S_IWUSR);
#endif
    }
    
    bool is_file(const char* path) {
        struct stat buf;
        stat(path, &buf);
        return S_ISREG(buf.st_mode);
    }
    
    bool is_dir(const char* path) {
        struct stat buf;
        stat(path, &buf);
        return S_ISDIR(buf.st_mode);
    }
    
    vector<string> extractImagePath(const string& targets) {

        vector<string> res;
        vector<String> fnJpg;
        vector<String> fnPng;
        vector<String> fnBmp;

        glob(targets + "/*.jpg", fnJpg);
        glob(targets + "/*.png", fnPng);
        glob(targets + "/*.bmp", fnBmp);

        for (auto jpgImage : fnJpg) {
            res.push_back(string(jpgImage.c_str()));
        }
        for (auto pngImage : fnPng) {
            res.push_back(string(pngImage.c_str()));
        }
        for (auto bmpImage : fnBmp) {
            res.push_back(string(bmpImage.c_str()));
        }

        return res;
    }

    string getName(string file_path) {

        std::replace(file_path.begin(), file_path.end(), '\\', '/');

        auto offset = file_path.find_last_of('/');

        return file_path.substr(offset + 1);
    }

    string getParentDirectory(string path) {
        std::replace(path.begin(), path.end(), '\\', '/');
        auto offset = path.find_last_of('/');

        return path.substr(0, offset);
    }

    string set_format(string of_file_name_path, string to_new_format, bool use_separator) {
        int image_format_offset_begin = of_file_name_path.find_last_of(".");
        string image_format = of_file_name_path.substr(image_format_offset_begin);
        int image_format_offset_end = image_format.length() + image_format_offset_begin;

        return of_file_name_path.replace(image_format_offset_begin, image_format_offset_end,
                                         (use_separator ? "." : "") + to_new_format);
    }

    void loadFromJSON(const string& target, vector<Features> &features, Mat &labels) {

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

    void saveFeaturesJSON(const vector<Features> &features, const vector<string>& names, const string &saveFile) {

        vector<char> tmp(saveFile.begin(), saveFile.end());

        const auto directory = string(dirname(tmp.data()));

        portable_mkdir(directory.data());

        ifstream iJson(saveFile, fstream::in);

        std::string rawJson((std::istreambuf_iterator<char>(iJson)), std::istreambuf_iterator<char>());

        iJson.close();

        Document doc;

        doc.Parse(rawJson.data());

        if (!doc.IsObject()) {
            doc.SetObject(); // It's necessary else the doc will start as "null"
            doc.AddMember("features", kArrayType, doc.GetAllocator());
        }

        assert(doc.HasMember("features"));
        assert(doc["features"].IsArray());

        for (int i = 0; i < features.size(); ++i) {

            const Features ft = features[i];

            const string c_name = set_format(getName(names[i]), "", false);

            Value obj(kObjectType);

            obj.AddMember("class", Value(ft._class), doc.GetAllocator());

            Value sample;
            // Do not move, do not change or I'll cut your hand
            sample.SetString(StringRef(c_name.data()), static_cast<unsigned>(c_name.length()), doc.GetAllocator());

            obj.AddMember("sample", sample, doc.GetAllocator());
            obj.AddMember("label", Value(ft.label), doc.GetAllocator());
            obj.AddMember("id", Value(ft.id), doc.GetAllocator());
            obj.AddMember("contrast", Value(ft.contrast), doc.GetAllocator());
            obj.AddMember("avgGreyValue", Value(ft.averageGreyValue), doc.GetAllocator());
            Value rgb(kArrayType);
            for (const double chVal : ft.averageRGBValues.val) rgb.PushBack(chVal, doc.GetAllocator());
            obj.AddMember("avgRGBValues", rgb, doc.GetAllocator());
            obj.AddMember("skewness", Value(ft.skewness), doc.GetAllocator());
            obj.AddMember("energy", Value(ft.energy), doc.GetAllocator());
            obj.AddMember("entropy", Value(ft.entropy), doc.GetAllocator());
            Value hog(kArrayType);
            for (const float descriptor : ft.hogDescriptors.row(0)) hog.PushBack(descriptor, doc.GetAllocator());
            obj.AddMember("hog", hog, doc.GetAllocator());

            doc["features"].PushBack(obj, doc.GetAllocator());

            imwrite(directory + "/" + c_name + "_L" + to_string(ft.label) + "_" + to_string(ft.id) + ".bmp",
                    ft.candidate);
        }

        StringBuffer buffer;
        PrettyWriter<StringBuffer> writer(buffer);
        doc.Accept(writer);

        ofstream oJson(saveFile, fstream::out);

        oJson << buffer.GetString();

        oJson.close();
    }

    Configuration loadProgramConfiguration(const string& target) {

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
            assert(config["thresholds"]["primary"]["density"].IsDouble());
            assert(config["thresholds"]["primary"].HasMember("variance"));
            assert(config["thresholds"]["primary"]["variance"].IsDouble());
            assert(config["thresholds"]["primary"].HasMember("gauss"));
            assert(config["thresholds"]["primary"]["gauss"].IsDouble());
            assert(config["thresholds"]["primary"].HasMember("minGreyRatio"));
            assert(config["thresholds"]["primary"]["minGreyRatio"].IsDouble());
            assert(config["thresholds"]["primary"].HasMember("maxGreyRatio"));
            assert(config["thresholds"]["primary"]["maxGreyRatio"].IsDouble());
            assert(config["thresholds"]["primary"].HasMember("minGreenRatio"));
            assert(config["thresholds"]["primary"]["minGreenRatio"].IsDouble());

            primary.density = config["thresholds"]["primary"]["density"].GetDouble();
            primary.variance = config["thresholds"]["primary"]["variance"].GetDouble();
            primary.gauss = config["thresholds"]["primary"]["gauss"].GetDouble();
            primary.minGreyRatio = config["thresholds"]["primary"]["minGreyRatio"].GetDouble();
            primary.maxGreyRatio = config["thresholds"]["primary"]["maxGreyRatio"].GetDouble();
            primary.minGreenRatio = config["thresholds"]["primary"]["minGreenRatio"].GetDouble();

            assert(config["thresholds"].HasMember("secondary"));
            assert(config["thresholds"]["secondary"].IsObject());

            assert(config["thresholds"]["secondary"].HasMember("density"));
            assert(config["thresholds"]["secondary"]["density"].IsDouble());
            assert(config["thresholds"]["secondary"].HasMember("variance"));
            assert(config["thresholds"]["secondary"]["variance"].IsDouble());
            assert(config["thresholds"]["secondary"].HasMember("gauss"));
            assert(config["thresholds"]["secondary"]["gauss"].IsDouble());
            assert(config["thresholds"]["secondary"].HasMember("minGreyRatio"));
            assert(config["thresholds"]["secondary"]["minGreyRatio"].IsDouble());
            assert(config["thresholds"]["secondary"].HasMember("maxGreyRatio"));
            assert(config["thresholds"]["secondary"]["maxGreyRatio"].IsDouble());
            assert(config["thresholds"]["secondary"].HasMember("minGreenRatio"));
            assert(config["thresholds"]["secondary"]["minGreenRatio"].IsDouble());

            secondary.density = config["thresholds"]["secondary"]["density"].GetDouble();
            secondary.variance = config["thresholds"]["secondary"]["variance"].GetDouble();
            secondary.gauss = config["thresholds"]["secondary"]["gauss"].GetDouble();
            secondary.minGreyRatio = config["thresholds"]["secondary"]["minGreyRatio"].GetDouble();
            secondary.maxGreyRatio = config["thresholds"]["secondary"]["maxGreyRatio"].GetDouble();
            secondary.minGreenRatio = config["thresholds"]["secondary"]["minGreenRatio"].GetDouble();

        } else {
            cerr << "Program configuration is missing. Check it's existence or create a new config.json"
                 << " under the " << target << " folder inside the program directory. " << endl;

            exit(-3);
        }

        return Configuration{offsets, primary, secondary};
    }

    void showElaborationStatusToTheUser(string showingWindowTitle, Mat processedImage){
        imshow(showingWindowTitle, processedImage);
        waitKey(1);
        char response = 'n';
        while(response != 'c' && response != 'C') {
            cout << "Type 'c' to continue" << endl;
            cin >> response;
        }
    }

    void showElaborationStatusToTheUser(const vector<Features> candidatesFeatures){
        for(auto features : candidatesFeatures){
            imshow("candidate: " + std::to_string(features.label), features.candidate);
            waitKey(1);
        }
        char response = 'n';
        while(response != 'c' && response != 'C') {
            cout << "Type 'c' to continue" << endl;
            cin >> response;
        }
    }

    void showElaborationStatusToTheUser(const vector<SuperPixel> superPixels){
        for(auto sp : superPixels){
            imshow("candidate: " + std::to_string(sp.label), sp.selection);
            waitKey(1);
        }
        char response = 'n';
        while(response != 'c' && response != 'C') {
            cout << "Type 'c' to continue" << endl;
            cin >> response;
        }
    }

    bool exists (const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }
}