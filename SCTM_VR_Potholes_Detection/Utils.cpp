#include <algorithm>
#include <sys/stat.h>
#include "Utils.h"

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

string set_format(string of_file_name_path, string to_new_format, bool use_separator) {
    int image_format_offset_begin = of_file_name_path.find_last_of(".");
    string image_format = of_file_name_path.substr(image_format_offset_begin);
    int image_format_offset_end = image_format.length() + image_format_offset_begin;

    return of_file_name_path.replace(image_format_offset_begin, image_format_offset_end,
                              (use_separator ? "." : "") + to_new_format);
}

void CSVTokenizer(const string str, const char delimiter, vector<string> &tokens){

//    remove_if(str.begin(), str.end(), ' '), str.end();

    std::istringstream iss(str);
    string token;
    while(std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
}

Features objectify(const vector<string> &tokens, Mat &labels) {

    Features ft;

    for (const auto &token : tokens) {
        vector<string> t;

        CSVTokenizer(token, ':', t);

        if (t[0] == "Label") {
            labels.push_back(stoi(t[1]));
        } else if (t[0] == "Contrast") {
            ft.contrast = stof(t[1]);
        } else if (t[0] == "Skewness") {
            ft.skewness = stof(t[1]);
        } else if (t[0] == "AvgGreyVal") {
            ft.averageGreyValue = stof(t[1]);
        } else if (t[0] == "Energy") {
            ft.energy = stof(t[1]);
        } else if (t[0] == "Entropy") {
            ft.entropy = stof(t[1]);
        }

//        else if (t[0] == "Histogram") {
//            string comma_separated_floats = t[1].substr(1, t[1].length()-2);
//            vector<string> values;
//
//            CSVTokenizer(comma_separated_floats, ',', values);
//
//            ft.histogram = Mat(1, (int) values.size(), CV_32FC1);
//
//            for (int i = 0; i < values.size(); ++i){
//                ft.histogram.at<float>(0, i) = stof(values[i]);
//            }
//        }
    }

    return ft;
}

void loadFromCSV(const string target, vector<Features> &ft, Mat &labels) {

    ifstream csv(target);

    if (csv.is_open()) {
        string line;

        while (std::getline(csv, line)) {
            vector<string> tokens;
            CSVTokenizer(line, ';', tokens);
            auto f = objectify(tokens, labels);

            ft.push_back(f);

//            cout << " Contrast:"   << f.contrast;
//            cout << " Skewness:"   << f.skewness;
//            cout << " AvgGreyVal:" << f.averageGreyValue;
//            cout << " Energy:"     << f.energy;
//            cout << " Entropy:"    << f.entropy;
//            cout << " Histogram:"  << f.histogram << endl;

        }
        csv.close();
    } else cout << "Unable to open file";

    cout << labels.rows << " " << ft.size() << endl;

}

string extractFileName(string file_path, const string sep = "/") {

    std::replace(file_path.begin(), file_path.end(), '\\', '/');

    auto offset = file_path.find_last_of(sep);

    return file_path.substr(offset+1);
}

void saveFeatures(const vector<Features> &ft, const string saveDirectory, const string imgName, const string saveFile) {

    auto candidate = extractFileName(imgName);

    portable_mkdir(("../" + saveDirectory).data());

    ofstream save_file;
    save_file.open ("../" + saveDirectory + "/" + saveFile + ".csv", fstream::in | fstream::out | fstream::app);

    for (int i = 0; i < ft.size(); ++i) {
        auto f = ft[i];

        string c_name = "../data/"+ set_format(candidate, "", false) + "_" + to_string(i) + ".bmp";

        cout << "Saving candidate " << c_name << endl;

        save_file << "Label:" << -1 << ";";
        save_file << "Candidate:"  << c_name << ";";
        save_file << "Contrast:"   << f.contrast << ";";
        save_file << "Skewness:"   << f.skewness << ";";
        save_file << "AvgGreyVal:" << f.averageGreyValue << ";";
        save_file << "Energy:"     << f.energy << ";";
        save_file << "Entropy:" << f.entropy << endl; //<< ";";
//        save_file << "Histogram:"  << f.histogram << endl;

//        imshow(c_name, f.candidate);
//        waitKey();
        imwrite(c_name, f.candidate);

    }

    save_file.close();
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