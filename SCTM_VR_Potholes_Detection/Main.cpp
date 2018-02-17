#include <cstdio>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <iterator>

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace cv::ximgproc;

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

	String img_path = "D:\\Xander_C\\Downloads\\test_rec1.jpg";
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

double GaussianEllipseFunction3D(cv::Point P, cv::Point O = cv::Point(0.0, 0.0), double SigmaX = 1.0, double SigmaY = 1.0, double A = 1.0, double Theta = 0.0) {

	double a = cos(Theta)*cos(Theta) / (2 * SigmaX*SigmaX) + sin(Theta)*sin(Theta) / (2 * SigmaY*SigmaY);
	double b = -sin(2 * Theta) / (2 * SigmaX*SigmaX) + sin(2 * Theta) / (2 * SigmaY*SigmaY);
	double c = sin(Theta)*sin(Theta) / (2 * SigmaX*SigmaX) + cos(Theta)*cos(Theta) / (2 * SigmaY*SigmaY);

	double x = a * (P.x - O.x)*(P.x - O.x) + 2 * b*(P.x - O.x)*(P.y - O.y) + c * (P.y - O.y)*(P.y - O.y);

	return A * exp(-x);

}

int PotholeSegmentation() {

	const double V_offset = 1.00;
	const double H_offset = 0.00;
	const double Cutline_offset = 0.60;
	const double VP_offset_X = 0.50;
	const double VP_offset_Y = 0.55;
	const int W = 640;
	const int H = 480;

	const double Gauss_RoadThreshold = 0.5999;

	String img_path = "D:\\Xander_C\\Downloads\\test_rec8.jpg";

	Size scale(W, H);
	Point2d translation_((double) -W / 2.0, (double) -H / 1.0), shrink_(3.0 / (double) W, 5.0 / (double) H);

	Mat src = imread(img_path, IMREAD_COLOR), tmp, imgCIELab, contour, labels;

	resize(src, src, scale);

	// Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
	GaussianBlur(src, tmp, Size(3, 3), 0.0);

	// Switch spazio colore da RGB CieLAB
	cvtColor(tmp, imgCIELab, COLOR_BGR2Lab);

	// Linear Spectral Clustering
	Ptr<SuperpixelLSC> superpixels = cv::ximgproc::createSuperpixelLSC(imgCIELab, 64);
	//Ptr<SuperpixelSLIC> superpixelSegmentation = cv::ximgproc::createSuperpixelSLIC(img1, SLIC::MSLIC, 32, 50.0);

	superpixels->iterate(12);

	superpixels->getLabelContourMask(contour);

	superpixels->getLabels(labels);

	int N_SP = superpixels->getNumberOfSuperpixels(); cout << N_SP << endl;

	src.copyTo(tmp);

	tmp.setTo(Scalar(0, 0, 0), contour);
	
	imshow("Contours", tmp);

	Mat out, mask, res; 
	src.copyTo(out); src.copyTo(mask); mask.setTo(Scalar(255, 255, 255));

	for (int l = 0; l < N_SP; ++l) {

		Mat1b LabelMask = (labels == l);

		// How to evaluate the thresholds in order to separate road super pixels?
		// Or directly identify RoI where a pothole willl be more likely detected?
		// ...
		// Possibilities => Gaussian 3D function

		vector<cv::Point> PixelsLabel;
		cv::Point SuperPixel(0.0, 0.0);

		findNonZero(LabelMask, PixelsLabel);
		for (auto p : PixelsLabel) SuperPixel += p;
		SuperPixel /= (double)PixelsLabel.size();
		
		Point translated_((SuperPixel.x + translation_.x), (SuperPixel.y + translation_.y));
		Point shrinked_(translated_.x*shrink_.x, translated_.x*shrink_.y);

		//cout << "Superpixel center @ " << SuperPixel << ", translated @ " << translated_ << ", shrinked @ " << shrinked_ << endl;

		auto gVal = GaussianEllipseFunction3D(shrinked_);

		printf("Superpixel %d =====>\t %2f\n", l, gVal);

		bool isRoad = gVal > Gauss_RoadThreshold;
		bool isOverHorizon = SuperPixel.y < (1 - Cutline_offset)*H;

		Scalar mean_color_value = isRoad && !isOverHorizon ? mean(src, LabelMask) : Scalar(0, 0, 0);
		Scalar color_mask_value = isRoad && !isOverHorizon ? Scalar(255, 255, 255) : Scalar(0, 0, 0);

		out.setTo(mean_color_value, LabelMask);
		mask.setTo(color_mask_value, LabelMask);
	}

	imshow("Segmentation", out);

	Mat outMask;

	// Dilate to clean possible small black dots into the image "center"
	auto dilateElem = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	// Remove smal white dots outside the image "center" 
	auto erodeElem = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));

	dilate(mask, outMask, dilateElem);
	erode(outMask, outMask, erodeElem);
	
	imshow("Mask", outMask);

	src.copyTo(res, outMask);

	imshow("Result", res);

	// Cut the image in order to resize it to the smalled square/rectangle possible

	waitKey();

	return 1;
}

int main(int argc, char*argv[]) {

	//return resize_all_in("D:/Xander_C/Downloads/DatasetNostro/Positivi/", "D:/Xander_C/Downloads/DatasetNostro/Positivi/imgs", 1920, 1080);

	//return MyCascadeClassifier();

	return PotholeSegmentation();
}