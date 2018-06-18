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

string extractFileName(string file_path, const string sep = "/") {

    std::replace(file_path.begin(), file_path.end(), '\\', '/');

    auto offset = file_path.find_last_of(sep);

    return file_path.substr(offset+1);
}

string set_format(string of_file_name_path, string to_new_format, bool use_separator) {
    int image_format_offset_begin = of_file_name_path.find_last_of(".");
    string image_format = of_file_name_path.substr(image_format_offset_begin);
    int image_format_offset_end = image_format.length() + image_format_offset_begin;

    return of_file_name_path.replace(image_format_offset_begin, image_format_offset_end,
                              (use_separator ? "." : "") + to_new_format);
}

vector<string> CSVTokenizer(const string str, const char delimiter){

    string tmp(str);
    auto end_pos = remove(tmp.begin(), tmp.end(), ' ');
    tmp.erase(end_pos, tmp.end());

    vector<string> tokens;

    std::istringstream iss(str);
    string token;
    while(std::getline(iss, token, delimiter)) tokens.push_back(token);

    return tokens;
}

Features objectify(const vector<string> &headers, const vector<string> &tokens, Mat &labels) {

    Features ft;

    for (int i = 0; i < tokens.size(); ++i) {
        const string header = headers[i];

        if (header == "class") {
            labels.push_back(stoi(tokens[i]));
        } else if (header == "contrast") {
            ft.contrast = stof(tokens[i]);
        } else if (header == "skewness") {
            ft.skewness = stof(tokens[i]);
        } else if (header == "avggreyval") {
            ft.averageGreyValue = stof(tokens[i]);
        } else if (header == "energy") {
            ft.energy = stof(tokens[i]);
        } else if (header == "entropy") {
            ft.entropy = stof(tokens[i]);
        }
    }

    return ft;
}

void loadFromCSV(const string target, vector<Features> &ft, Mat &labels) {

    ifstream csv(target);

    if (csv.is_open()) {

        cout << "Opened file " << target << endl;

        string line;
        // Get Headers;
        std::getline(csv, line);
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        // Do something with headers ...
        vector<string> headers = CSVTokenizer(line, ',');

        while (std::getline(csv, line)) {
            vector<string> tokens = CSVTokenizer(line, ',');
            auto f = objectify(headers, tokens, labels);

            ft.push_back(f);

        }
        csv.close();
    } else {
        cerr << "Unable to open file " << target << endl;
    }

    cout << labels.rows << " " << ft.size() << endl;

}

bool checkExistence(const string &target) {

    ifstream csv(target);

    if (csv.is_open()) {
        string line;
        std::getline(csv, line);

        // Normalize the string.
        auto end_pos = remove(line.begin(), line.end(), ' ');
        line.erase(end_pos, line.end());
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);

        cout << line << endl;
        if (line == "class,candidate,contrast,skewness,avggreyval,energy,entropy") {
            return true;
        }

        csv.close();
    } else {
        cerr << "Unable to open file " << target << ". It may not exist!" << endl;
    }

    return false;
}

void saveFeatures(const vector<Features> &features, const string saveDirectory, const vector<string> names,
                  const string saveFile) {

    bool doesNotExist = !checkExistence("../" + saveDirectory + "/" + saveFile + ".csv");

    portable_mkdir(("../" + saveDirectory).data());

    ofstream csv("../" + saveDirectory + "/" + saveFile + ".csv", fstream::in | fstream::out | fstream::app);

    if(csv.is_open()) {

        if (doesNotExist) {
            csv << "Class,Candidate,Contrast,Skewness,AvgGreyVal,Energy,Entropy" << endl;
        }

        for (int i = 0; i < features.size(); ++i) {

            auto f = features[i];
            auto candidate = extractFileName(names[i]);

            string c_name = set_format(candidate, "", false) + "_L" + to_string(f.SPLabel);

            cout << "Saving candidate " << c_name << endl;

            csv << -1 << ",";
            csv << c_name << ",";
            csv << f.contrast << ",";
            csv << f.skewness << ",";
            csv << f.averageGreyValue << ",";
            csv << f.energy << ",";
            csv << f.entropy << endl;

            imwrite("../" + saveDirectory +"/" + c_name + ".bmp", f.candidate);

        }

        csv.close();
    } else {
        cerr << "Ops, we got a problem opening the feature file!" << endl;
    }
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