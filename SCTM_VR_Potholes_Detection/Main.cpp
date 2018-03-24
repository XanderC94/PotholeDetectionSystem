#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Segmentation.h"
#include "HistogramElaboration.h"

using namespace cv;
using namespace std;

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

int main(int argc, char*argv[]) {

	if (argc < 2) {
		return 0;
	} else {

		auto centroids = vector<Point>();
		auto candidates = vector<Mat>();
		Mat imgGrayScale;

		auto candidate_size = Size(128, 128);

		Mat src = imread(argv[1], IMREAD_COLOR), tmp;

		/*
			The src image will be:
			1. Resized to 640 * 480
			2. Cropped under the Horizon Line
			3. Blurred with a Gaussian Function

			in Candidates will be placed the set on centroids that survived segmentation
		*/
		PotholeSegmentation(src, centroids, 32, 0.65, 0.8);

		// Candidate Extraction
		for (auto c : centroids) {

			auto tlc_x = c.x - candidate_size.width*0.5;
			auto tlc_y = c.y - candidate_size.height*0.5;
			auto brc_x = c.x + candidate_size.width*0.5;
			auto brc_y = c.y + candidate_size.height*0.5;

			auto tlc = Point(tlc_x < 0 ? 0 : tlc_x, tlc_y < 0 ? 0 : tlc_y);
			auto brc = Point(brc_x > src.cols - 1 ? src.cols : brc_x, brc_y > src.rows - 1 ? src.rows : brc_y);

			auto candidate = src(Rect(tlc, brc));

			candidates.push_back(candidate);

			// Switch spazio di colore da RGB a GreayScale
			cvtColor(candidate, imgGrayScale, CV_BGR2GRAY);
			imshow("Grey scale", imgGrayScale);

			ExtractHistograms(imgGrayScale);
			waitKey();
		}

		return 1;
	}
}