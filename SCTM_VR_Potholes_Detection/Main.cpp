#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "Segmentation.h"
#include "FeaturesExtraction.h"

using namespace cv;
using namespace std;

enum CLASSES {
	NORMAL, // 0
	POTHOLE	// 1
};

int main(int argc, char*argv[]) {

    //Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
		return 0;
	} else {

		auto centroids = vector<Point>();

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
        auto features = extractFeatures(src, centroids, candidate_size);

        //Calculate and Print the execution time
        timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
        cout << "Times passed in seconds: " << timeElapsed << endl;

        waitKey();
        return 1;
	}
}