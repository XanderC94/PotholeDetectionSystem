#include <cstdio>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <iterator>

using namespace cv;
using namespace std;
using namespace cv::ml;

const string data_root_path = "D:/Xander_C/Downloads/sctm/";
const string train_home_directory = data_root_path + "Dataset/";
const string classifier64 = train_home_directory + "Classifier64/cascade.xml";
const string classifier32 = train_home_directory + "Classifier32/cascade.xml";
const string classifier64LBP = train_home_directory + "Classifier64LBP/cascade.xml";
const string negative_directory = train_home_directory + "Train/Negative/Images/*";
const string positive_directory = train_home_directory + "Train/Positive/Images/*";
const string negative_training_set_directory = train_home_directory + "Train/Negative/Masks/*";
const string positive_training_set_directory = train_home_directory + "Train/Positive/Masks/*";

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

void set_format(string & of_file_name_path, string to_new_format, bool use_separator = true) {
	int image_format_offset_begin = of_file_name_path.find_last_of(".");
	string image_format = of_file_name_path.substr(image_format_offset_begin);
	int image_format_offset_end = image_format.length() + image_format_offset_begin;

	of_file_name_path.replace(image_format_offset_begin, image_format_offset_end, (use_separator ? "." : "") + to_new_format);
}

void load_from_directory(const string & directory, vector<string> & ids, vector<Mat> & set, Mat & labels, int label = -1, int image_type = IMREAD_COLOR) {

	vector<String> fn;
	glob(directory, fn);

	cout << "Found " << fn.size() << " images..." << endl;

	for (auto &i : fn) {
		string file_name = i;

		Mat img = imread(file_name, image_type);

		if (img.empty())
		{
			cerr << "invalid image: " << file_name << endl;
			continue;
		}
		else 
		{
			cout << "Loaded image " << file_name << endl;
			set.push_back(img);
			labels.push_back(label);
			ids.push_back(file_name.substr(file_name.find_last_of("\\")+1));
		}
	}
}

int resize_all_in(const string parent, const string folder, const int width = 1280, const int height = 720) {

	vector<String> fn;
	glob(folder + "/*", fn);

	cout << "Found " << fn.size() << " images..." << endl;

	for (int i = 0; i < fn.size(); i++)
	{
		string file_name = fn[i];
		Mat img = imread(file_name, IMREAD_COLOR);

		if (img.empty())
		{
			cerr << "invalid image: " << file_name << endl;
			continue;
		}
		else
		{
			cout << "Loaded image " << file_name << endl;

			Mat dst;
			resize(img, dst, Size(width, height));

			imwrite(parent + "/scaled/" + file_name.substr(file_name.find_last_of("\\")), dst);
		}
	}

	return 1;
}

int MyCascadeClassifier() {

	const double V_offset = 1.00;
	const double H_offset = 0.00;
	const double Cutline_offset = 0.60;
	const double VP_offset_X = 0.50;
	const double VP_offset_Y = 0.55;

	String img_path = "D:\\Xander_C\\Downloads\\test_rec7.jpg";
	//String img_path = "D:\\Xander_C\\Downloads\\sctm\\Dataset_Orig\\Train\\Positive\\Images\\G0011474.JPG";

	string cls = classifier64;

	vector<Rect> output;
	Size scale(640, 480);

	Mat img0 = imread(img_path, IMREAD_COLOR), img1;

	cvtColor(img0, img0, CV_RGB2GRAY);

	img0.convertTo(img1, CV_8U);

	resize(img1, img0, scale);

	/*
	
	Point left_roadside[1][3] = {
		{ Point(0, 0), Point(img0.cols*VP_offset_X, img0.rows*VP_offset_Y), Point(0, img0.rows*V_offset) }
	};

	Point right_roadside[1][3] = {
		{ Point(img0.cols*VP_offset_X, img0.rows*VP_offset_Y), Point(img0.cols - 1, img0.rows*V_offset), Point(img0.cols - 1, 0) }
	};

	fillConvexPoly(img0, left_roadside[0], 3, Scalar(0, 0, 0));

	fillConvexPoly(img0, right_roadside[0], 3, Scalar(0, 0, 0));
	
	*/

	img0 = img0(Rect(Point(0, img0.rows*Cutline_offset), Point(img0.cols - 1, img0.rows - 1)));

	CascadeClassifier cascade;
	bool b = cascade.load(cls);

	clog << "Loaded casced classifier @" << cls << " : " << (b == 0 ? "False" : "True") << endl;

	cascade.detectMultiScale(img0, output);

	clog << "Detected " << output.size() << " objects as potholes..." << endl;

	for (auto &r : output) {
		rectangle(img0, r, Scalar(255, 0, 0));
	}

	imshow("Pothole Detection", img0);

	waitKey();

	return 1;
}

int main(int argc, char*argv[]) {

	//return resize_all_in("D:/Xander_C/Downloads/DatasetNostro/Positivi/", "D:/Xander_C/Downloads/DatasetNostro/Positivi/imgs", 1920, 1080);

	return MyCascadeClassifier();
}