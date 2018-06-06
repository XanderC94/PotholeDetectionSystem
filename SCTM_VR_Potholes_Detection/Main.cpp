#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include "Segmentation.h"
#include "FeaturesExtraction.h"
#include "SVM.h"

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
        vector<int> labels;

        Mat imgBlurred;
        auto candidate_size = Size(64, 64);

        /*---------------------------------Load image------------------------*/
		Mat src = imread(argv[1], IMREAD_COLOR), tmp;

		if (argc <= 3) {
            isTraining = true;
		}

        /*-------------------------- Segmentation Phase --------------------*/

        cout << "Preprocessing... ";
        // Apply gaussian blur in order to smooth edges and gaining cleaner superpixels
        GaussianBlur(src, src, Size(5, 5), 0.0);
//        imshow("Blurred Image", imgBlurred);

        /*
		*	The src image will be:
		*	1. Resized to 640 * 480
		*	2. Cropped under the Horizon Line
        *    3. Segmented with superpixeling
        *    4. Superpixels in the RoI is selected using Analytic Rect Function
		*	(in Candidates will be placed the set on centroids that survived segmentation)
		*/
        cout << "Finished." << endl;
        Offsets offsets = {0.65, 0.15, 0.8};
        ExtractionThresholds threshold = {0.80, 0.30, 0.60};
        int superPixelEdge = 32;


        PotholeSegmentation(src, centroids, superPixelEdge, threshold, offsets);
        cout << "Finished." << endl;
        cout << "Found " << centroids.size() << " candidates." << endl;

        /*--------------------------------- End Segmentation Phase ------------------------------*/

        /*--------------------------------- Feature Extraction Phase ------------------------------*/

        cout << "Feature Extraction... " << endl;
        auto features = extractFeatures(src, centroids, candidate_size);
        cout << "Finished." << endl;

        if (isTraining) {
            printf("Starting Training...\n");
            for (int i = 0; i < features.size(); ++i) {
                labels.push_back(1);
            }
            Training(features, labels, 100, "../svm_trained_model.yml");
        } else {
            auto labels = Classifier(features, 100, "../svm_trained_model.yml");
        }

        //Calculate and Print the execution time
        timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
        cout << "Times passed in seconds: " << timeElapsed << endl;

        waitKey();
        return 1;
	}
}