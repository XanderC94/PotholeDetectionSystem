#include <stdio.h>
#include <stdlib.h>
#include <opencv2/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <iterator>

using namespace cv;
using namespace std;
using namespace cv::ml;

const string data_root_path = "D:\\Xander_C\\Downloads\\sctm\\";
const string algorithm_save_location = data_root_path + "trained_bayes.xml";
const string training_set_directory = data_root_path + "Dataset_Mix_Test\\Train\\";
const string negative_directory = training_set_directory + "Negative\\Images\\*";
const string positive_directory = training_set_directory + "Positive\\Images\\*";
const string negative_training_set_directory = training_set_directory + "Negative\\Masks\\*";
const string positive_training_set_directory = training_set_directory + "Positive\\Masks\\*";

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

void hold() {
	char * c = "a";
	int i = scanf_s(c);
}

void set_format(string & of_file_name_path, string to_new_format, bool use_separator = true) {
	size_t image_format_offset_begin = of_file_name_path.find_last_of(".");
	string image_format = of_file_name_path.substr(image_format_offset_begin);
	size_t image_format_offset_end = image_format.length() + image_format_offset_begin;

	of_file_name_path.replace(image_format_offset_begin, image_format_offset_end, (use_separator ? "." : "") + to_new_format);
}

void load_from_directory(const string & directory, vector<string> & ids, vector<Mat> & set, Mat & labels = Mat(), int label = -1, int image_type = IMREAD_COLOR) {

	vector<String> fn;
	glob(directory, fn);

	cout << "Found " << fn.size() << " images..." << endl;

	for (size_t i = 0; i < fn.size(); i++)
	{
		string file_name = fn[i];

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

void create_training_set_by_row(const vector<Mat> & set, Mat & training_set) {

	for (size_t i = 0; i < set.size(); ++i) {
		Mat img;
		set[i].convertTo(img, CV_32FC1);
		training_set.push_back(img.reshape(0, 1));
	}
}

int resize_all_in(const string folder, const int width = 1280, const int height = 720) {

	vector<String> fn;
	glob(folder + "\\*", fn);

	cout << "Found " << fn.size() << " images..." << endl;

	for (size_t i = 0; i < fn.size(); i++)
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

			imwrite(folder + "\\scaled\\" + file_name.substr(file_name.find_last_of("\\")), dst);
		}
	}

	return 1;
}

Mat get_hogdescriptor_visual_image(const Mat& or_img, const vector< float>& dscr, const Size win, const Size cell, const int bins = 9, const int scale = 1, const double grad_viz = 5, const bool isGrayscale = false, bool onlyMaxGrad = true)
{
	Mat visual_image;
	if (scale == 1) {
		resize(or_img, visual_image, Size(or_img.cols*scale, or_img.rows*scale));
	}
	
	if (isGrayscale) {
		cvtColor(visual_image, visual_image, CV_GRAY2BGR);
	}
	
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float)bins;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = win.width / cell.width;
	int cells_in_y_dir = win.height / cell.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y< cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x< cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[bins];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin< bins; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx< blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky< blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr< 4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin< bins; bin++)
				{
					float gradientStrength = dscr[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (int celly = 0; celly< cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx< cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin< bins; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}


	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

	// draw cells
	for (int celly = 0; celly< cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx< cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cell.width;
			int drawY = celly * cell.height;

			int mx = drawX + cell.width / 2;
			int my = drawY + cell.height / 2;

			rectangle(visual_image,
				Point(drawX*scale, drawY*scale),
				Point((drawX + cell.width)*scale,
				(drawY + cell.height)*scale),
				CV_RGB(100, 100, 100),
				1);

			if (onlyMaxGrad) {
				double maxGrad = 0.0; int idx = 0;
				
				for (int bin = 0; bin < bins; bin++) {
					if (gradientStrengths[celly][cellx][bin] > maxGrad) {
						maxGrad = gradientStrengths[celly][cellx][bin];
						idx = bin;
					}
				}

				// no line to draw?
				if (maxGrad == 0)
					continue;

				float currRad = idx * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cell.width / 2;
				float scale = grad_viz; // just a visual_imagealization scale,
										  // to see the lines better

										  // compute line coordinates
				float x1 = mx - dirVecX * maxGrad * maxVecLen * scale;
				float y1 = my - dirVecY * maxGrad * maxVecLen * scale;
				float x2 = mx + dirVecX * maxGrad * maxVecLen * scale;
				float y2 = my + dirVecY * maxGrad * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					Point(x1*scale, y1*scale),
					Point(x2*scale, y2*scale),
					CV_RGB(0, 0, 255),
					1);
			}
			else {
				// draw in each cell all 9 gradient strengths
				for (int bin = 0; bin < bins; bin++)
				{
					float currentGradStrength = gradientStrengths[celly][cellx][bin];

					// no line to draw?
					if (currentGradStrength == 0)
						continue;

					float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

					float dirVecX = cos(currRad);
					float dirVecY = sin(currRad);
					float maxVecLen = cell.width / 2;
					float scale = grad_viz; // just a visual_imagealization scale,
											  // to see the lines better

											  // compute line coordinates
					float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
					float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
					float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
					float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

					// draw gradient visual_imagealization
					line(visual_image,
						Point(x1*scale, y1*scale),
						Point(x2*scale, y2*scale),
						CV_RGB(0, 0, 255),
						1);

				} // for (all bins)
			}

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y< cells_in_y_dir; y++)
	{
		for (int x = 0; x< cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;

}

void HOG() {

	vector<Mat> set;
	Mat classes;
	Mat training_set;
	vector<string> ids;

	load_from_directory(positive_training_set_directory, ids, set, classes, CLASSES::POTHOLE);
	//load_from_directory(negative_training_set_directory, ids, set, classes, CLASSES::NORMAL);

	const Size window(640, 480), block(64, 64), block_stride(16, 16), cell(16, 16), stride(8, 8), padding(0, 0);
	const int bin = 9;
	cv::HOGDescriptor hog = HOGDescriptor(window, block, block_stride, cell, bin);
	vector<Mat> descriptors;

	for (size_t i = 0; i < set.size(); ++i) {

		vector<float> ft;
		vector<Point> locs;
		Mat image = set[i];
		//Mat edges;

		cvtColor(image, image, CV_RGB2GRAY);

		/*
		string canny_save_loc = training_set_directory + (classes.at<int>(i) == 1 ? "Positive\\" : "Negative\\") + "Canny\\" + ids[i];
		Canny(image, edges, 15, 70);
		imwrite(canny_save_loc, edges);
		*/

		cout << "Resizing image number " << i << " @ " << window << endl;
		resize(image, image, window);

		cout << "Computing HOG Descriptor...";
		hog.compute(image, ft, stride, padding, locs);

		Mat d = get_hogdescriptor_visual_image(image, ft, window, cell);

		//descriptors.push_back(d);
		string dscr_save_loc = training_set_directory + (classes.at<int>(i) == 1 ? "Positive\\" : "Negative\\") + "HOG\\" + ids[i];
		cout << "Saving Descriptor @ " << dscr_save_loc << endl;

		imwrite(dscr_save_loc, d);
	}
}

void Bayes() {

	vector<Mat> set;
	Mat training_set, classes, tmp;
	vector<string> ids;

	load_from_directory(positive_training_set_directory, ids, set, classes, CLASSES::POTHOLE, IMREAD_GRAYSCALE);
	load_from_directory(negative_training_set_directory, ids, set, classes, CLASSES::NORMAL, IMREAD_GRAYSCALE);

	create_training_set_by_row(set, tmp);

	cout << tmp.size() << " --- " << classes.size() << endl;

	Ptr<NormalBayesClassifier> Bayes = NormalBayesClassifier::create();

	cout << "Starting training... ";
	
	tmp.convertTo(training_set, CV_32FC1);
	
	cout << training_set.type() << endl;

	Bayes->train(training_set, SampleTypes::ROW_SAMPLE, classes);

	cout << "Saved Trained Classifier @ " << algorithm_save_location << endl;

	Bayes->save(algorithm_save_location);
}

// All the values are reffered top-down for verticals and right to left for horizontals
void ApplyFixedRoadLinesAndHorizon(const double V_offset = 0.95, const double H_offset = 0.00, const double Cutline_offset = 0.50, const double VP_offset_X = 0.50, const double VP_offset_Y = 0.30) {
	
	vector<Mat> set;
	Mat classes;
	vector<string> ids;
	
	Scalar filling_color(0, 0, 0);
	Size window(360, 240);
	double blur_strength = 7.0;

	load_from_directory(positive_directory, ids, set, classes, CLASSES::POTHOLE);
	load_from_directory(negative_directory, ids, set, classes, CLASSES::NORMAL);

	for (size_t i = 0; i < set.size(); ++i) {
		
		Mat tmp0, tmp1;
		vector<Vec2f> lines;

		set[i].copyTo(tmp0);

		cvtColor(tmp0, tmp0, CV_RGB2GRAY);

		// Fill roadside with mask
		Point left_roadside[1][3] = {
			{Point(0, 0), Point(tmp0.cols*VP_offset_X, tmp0.rows*VP_offset_Y), Point(0, tmp0.rows*V_offset)}
		};

		Point right_roadside[1][3] = {
			{Point(tmp0.cols*VP_offset_X, tmp0.rows*VP_offset_Y), Point(tmp0.cols - 1, tmp0.rows*V_offset), Point(tmp0.cols - 1, 0)}
		};

		fillConvexPoly(tmp0, left_roadside[0], 3, filling_color);
		fillConvexPoly(tmp0, right_roadside[0], 3, filling_color);

		cout << "Cutting @ horizon..." << endl;

		tmp0 = tmp0(Rect(Point(0, tmp0.rows*Cutline_offset), Point(tmp0.cols - 1, tmp0.rows - 1)));	

		resize(tmp0, tmp1, window);
		
		string masks_save_location = training_set_directory + (classes.at<int>(i) == 1 ? "Positive\\" : "Negative\\") + "Masks\\" + ids[i];

		set_format(masks_save_location, "bmp");

		imwrite(masks_save_location, tmp1);

		cout << "Saved Roadmask @ " << masks_save_location << endl;

		/************************************************************************************************************************************/

		//Canny(tmp0, tmp1, 50, 150);

		threshold(tmp1, tmp1, 150, 255, THRESH_OTSU | THRESH_BINARY_INV);

		string bin_save_location = training_set_directory + (classes.at<int>(i) == 1 ? "Positive\\" : "Negative\\") + "Binaries\\" + ids[i];

		set_format(bin_save_location, "bmp");

		imwrite(bin_save_location, tmp1);

		cout << "Saved ths @ " << bin_save_location << endl;

		tmp0.release();
		tmp1.release();

		set[i].release();

		/************************************************************************************************************************************/
	}

	classes.release();
}

int main(int argc, char*argv[]) {
	
	ApplyFixedRoadLinesAndHorizon();

	Bayes();

	hold();

	return 1;
}