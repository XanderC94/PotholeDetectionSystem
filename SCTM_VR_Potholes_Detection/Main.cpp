#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include "Segmentation.h"
#include "FeaturesExtraction.h"
#include "SVM.h"
#include "Utils.h"

using namespace cv;
using namespace std;

using namespace cv::ml;

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

int main(int argc, char*argv[]) {

    //Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();
    bool isTraining = false;

    if (argc < 2) {
		return 0;
	} else {

		auto centroids = vector<Point>();

        auto candidate_size = Size(64, 64);

        /*---------------------------------Load image------------------------*/
		Mat src = imread(argv[1], IMREAD_COLOR), tmp;

		if (argc <= 3) {
            isTraining = true;
		}

        /*-------------------------- First Segmentation Phase --------------------*/

        cout << "Preprocessing... ";
        // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
        GaussianBlur(src, src, Size(5, 5), 0.0);

        /*
		*	The src image will be:
		*	1. Resized to 640 * 480
		*	2. Cropped under the Horizon Line
        *   3. Segmented with superpixeling
        *   4. Superpixels in the RoI is selected using Analytic Rect Function
		*	(in Candidates will be placed the set on centroids that survived segmentation)
		*/
        cout << "Finished." << endl;
        Offsets offsets = {0.65, 0.15, 0.8};
        ExtractionThresholds threshold = {0.80, 0.30, 0.60};
        int superPixelEdge = 32;

        cout << "Segmentation... " << endl;
        PotholeSegmentation(src, centroids, superPixelEdge, threshold, offsets);
        cout << "Finished." << endl;
        cout << "Found " << centroids.size() << " candidates." << endl;

        /*--------------------------------- End First Segmentation Phase ------------------------------*/

        /*--------------------------------- Feature Extraction Phase ------------------------------*/

        cout << "Feature Extraction -- Starting " << endl;
        auto features = extractFeatures(src, centroids, candidate_size);
        cout << "Feature Extraction -- Finished." << endl;

        /*--------------------------------- End Feature Extraction Phase ------------------------------*/


        /*--------------------------------- Training Or Classification Phase ------------------------------*/

        saveFeatures(features, "data", argv[1], "features");

        if (isTraining) {

            printf("Starting Training...\n");

            Mat labels(0, 0, CV_32SC1);
            vector<Features> candidates;
            loadFromCSV("../data/features.csv", candidates, labels);

            Training(features, labels, 100, "../data/svm_trained_model.yml");
        } else {
            Mat labels((int) features.size(), 1, CV_32SC1);
            Classifier(features, 100, "../data/svm_trained_model.yml", labels);
        }

        //Calculate and Print the execution time
        timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
        cout << "Times passed in seconds: " << timeElapsed << endl;

        return 1;
	}
}