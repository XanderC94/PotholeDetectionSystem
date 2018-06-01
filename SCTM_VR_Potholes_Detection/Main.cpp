#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "Segmentation.h"
#include "HistogramElaboration.h"
#include "MathUtils.h"

using namespace cv;
using namespace std;

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

typedef struct Feature {
    Point2d centroid;
//    Mat candidate;
    Mat histogram;
} Feature;

int main(int argc, char*argv[]) {

    //Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
		return 0;
	} else {

		auto centroids = vector<Point>();
		auto candidates = vector<Mat>();
        auto features = vector<Feature>();

        auto candidatesAverageGreyLevel = vector<double>();
        auto candidatesContrast = vector<double>();
        auto candidatesEntropy = vector<double>();
        auto candidatesSkewness = vector<double>();


        Mat candidateGrayScale;
        Mat imgBlurred;
        auto candidate_size = Size(64, 64);

        /*---------------------------------Load image------------------------*/
		Mat src = imread(argv[1], IMREAD_COLOR), tmp;


        /*-------------------------- Segmentation Phase --------------------*/

        // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
        GaussianBlur(src, src, Size(5, 5), 0.0);
        //imshow("Blurred Image", imgBlurred);

        /*
		*	The src image will be:
		*	1. Resized to 640 * 480
		*	2. Cropped under the Horizon Line
        *    3. Segmented with superpixeling
        *    4. Superpixels in the RoI is selected using Analytic Rect Function
		*	(in Candidates will be placed the set on centroids that survived segmentation)
		*/

        Offsets offsets = {0.60, 0.15, 0.8};

        int superPixelEdge = 32;
        PotholeSegmentation(src, centroids, superPixelEdge, defaultThresholds, offsets);

        cout << "Centroids Number: " << centroids.size() << endl;

        /*--------------------------------- End Segmentation Phase ------------------------------*/

        /*--------------------------------- Feature Extraction Phase ------------------------------*/
        /*
         * Each centroid of superpixels extracted in the previous phase
         *  1. Candidate will be converted to greyscale
         *  2. The histogram will be calculated
         *  3. Calculate the average gray value
         *  4. Calculate the contrast
         *  5. Calculate 3-order moments (is Skewness according to http://aishack.in/tutorials/image-moments/)
         *  6. Calculate Consistency
         *  7. Calculate Entropy
         *  8. Put the calculated features in a eigenvector
         *
         * */
        /*------------------------Candidate Extraction---------------------------*/
		for (auto c : centroids) {
            //tlc (top left corner) brc(bottom right corner)
			auto tlc_x = c.x - candidate_size.width*0.5;
			auto tlc_y = c.y - candidate_size.height*0.5;
			auto brc_x = c.x + candidate_size.width*0.5;
			auto brc_y = c.y + candidate_size.height*0.5;

			auto tlc = Point2d(tlc_x < 0 ? 0 : tlc_x, tlc_y < 0 ? 0 : tlc_y);
			auto brc = Point2d(brc_x > src.cols - 1 ? src.cols : brc_x, brc_y > src.rows - 1 ? src.rows : brc_y);

			auto candidate = src(Rect(tlc, brc));

			candidates.push_back(candidate);

			// Switch color-space from RGB to GreyScale
			cvtColor(candidate, candidateGrayScale, CV_BGR2GRAY);

            double contrast = 0.0;
            double entropy = 0.0;

            for (int i = 0; i < candidateGrayScale.rows; i++) {
                for (int j = 0; j < candidateGrayScale.cols; j++) {
                    contrast = contrast + pow((i - j), 2) * candidateGrayScale.at<uchar>(i, j);
                    entropy =
                            entropy + (candidateGrayScale.at<uchar>(i, j) * log10(candidateGrayScale.at<uchar>(i, j)));
                }
            }

            candidatesContrast.push_back(contrast);

            entropy = 0 - entropy;

            //candidatesEntropy.push_back(entropy);

            Scalar averageGreyValue = mean(candidateGrayScale);
            candidatesAverageGreyLevel.push_back(averageGreyValue[0]);
            //cout << "Valore Medio " << averageGreyValue[0] << endl;

            double skewness = calculateSkewnessGrayImage(candidate, averageGreyValue[0]);
            candidatesSkewness.push_back(skewness);
            //cout << "Skewness " << skewness << endl;

            auto c_name = "Candidate @ (" + to_string(c.x) + ", " + to_string(c.y) + ")";
			imshow(c_name, candidateGrayScale);

            cout << c_name << endl;
            cout <<
                 "Average Gray Value: " << averageGreyValue[0] <<
                 " Contrast: " << contrast <<
                 " Skeweness: " << skewness <<
                 " Entropy: " << entropy << endl;

			Mat histogram = ExtractHistograms(candidateGrayScale);

            features.push_back(Feature{c, histogram});
		}

        //Calculate and Print the execution time
        timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
        cout << "Times passed in seconds: " << timeElapsed << endl;

        waitKey();
        return 1;
	}
}