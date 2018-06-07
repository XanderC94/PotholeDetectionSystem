#include "Utils.h"

string set_format(string of_file_name_path, string to_new_format, bool use_separator) {
    int image_format_offset_begin = of_file_name_path.find_last_of(".");
    string image_format = of_file_name_path.substr(image_format_offset_begin);
    int image_format_offset_end = image_format.length() + image_format_offset_begin;

    return of_file_name_path.replace(image_format_offset_begin, image_format_offset_end,
                              (use_separator ? "." : "") + to_new_format);
}

void load_from_directory(const string &directory,
                         vector<string> &ids,
                         vector<Mat> &set,
                         Mat &labels,
                         int label,
                         int image_type) {

    vector<String> fn;
    glob(directory, fn);

    cout << "Found " << fn.size() << " images..." << endl;

    for (auto &i : fn) {
        string file_name = i;

        Mat img = imread(file_name, image_type);

        if (img.empty()) {
            cerr << "invalid image: " << file_name << endl;
            continue;
        } else {
            cout << "Loaded image " << file_name << endl;
            set.push_back(img);
            labels.push_back(label);
            ids.push_back(file_name.substr(file_name.find_last_of("\\") + 1));
        }
    }
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

string extractFileName(const string file_path, const string sep = "/") {

    auto offset = file_path.find_last_of(sep);

    return file_path.substr(offset+1);
}

void saveFeatures(const vector<Features> &ft, const string directory, const string parent, const string target) {

    auto candidate = extractFileName(parent);

    ofstream save_file;
    save_file.open ("../" + directory + "/" + target + ".csv", fstream::in | fstream::out | fstream::app);

    for (int i = 0; i < ft.size(); ++i) {
        auto f = ft[i];

        save_file << "Label: " << -1 << "; ";
        save_file << "Candidate: "  << "../" << set_format(candidate, "", false) << "_" << i << ".bmp" << "; ";
        save_file << "Contrast: "   << f.contrast << "; ";
        save_file << "Skewness: "   << f.skewness << "; ";
        save_file << "AvgGrayVal: " << f.averageGrayValue << "; ";
        save_file << "Energy: "     << f.energy << "; ";
        save_file << "Entropy: "    << f.entropy << "; " << endl;
//        save_file << "Histogram: "  << f.histogram << "; " << endl;

    }

    save_file.close();
}