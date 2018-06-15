#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "Segmentation.h"
#include "FeaturesExtraction.h"
#include "SVM.h"
#include "Utils.h"
#include "HOG.h"

using namespace cv;
using namespace std;

using namespace cv::ml;

vector<Features> preClassification (const string target) {

    auto superPixels = vector<SuperPixel>();

    auto candidate_size = Size(128, 128);

    /*---------------------------------Load image------------------------*/

    Mat src = imread(target, IMREAD_COLOR), tmp;

//    cvtColor(src, src, CV_RGB2BGR);
    /*
    *	The src image will be:
    *	1. Resized
    *	2. Cropped under the Horizon Line
    *   3. Segmented with superpixeling
    *   4. Superpixels in the RoI is selected using Analytic Rect Function
    *	(in Candidates will be placed the set on superPixels that survived segmentation)
    */

    RoadOffsets offsets = {
            .Horizon_Offset = 0.65,
            .SLine_X_Right_Offset = 0.0,
            .SLine_X_Left_Offset = 0.4,
            .SLine_Y_Offset = 0.8
    };

    ExtractionThresholds threshold = {
            .Density_Threshold = 0.60, // OK, do not change
            .Variance_Threshold = 0.3,
            .Gauss_RoadThreshold = 0.60,
            .colourRatioThresholdMin= 1.25,
            .colourRatioThresholdMax = 2.5
    };

    int superPixelEdge = 32;

    printThresholds(threshold);
    printOffsets(offsets);

    /*--------------------------------- Pre-Processing Phase ------------------------------*/

    cout << "Preprocessing... ";
    preprocessing(src, src, offsets.Horizon_Offset);
    cout << "Finished." << endl;

//    imshow("IMAGE" + target, src);
//    waitKey();

    /*--------------------------------- End Pre-Processing Phase ------------------------------*/

    /*--------------------------------- First Segmentation Phase ------------------------------*/

    cout << "Segmentation... " << endl;
    extractRegionsOfInterest(src, superPixels, superPixelEdge, threshold, offsets);
    cout << "Finished." << endl;

    cout << "Found " << superPixels.size() << " candidates." << endl;

    /*--------------------------------- End First Segmentation Phase ------------------------------*/

//    for (auto sp: superPixels) {
//        imshow("SP" + to_string(sp.label), sp.superPixelSelection);
//    }
//
//    waitKey();

    /*--------------------------------- Feature Extraction Phase ------------------------------*/

    cout << "Feature Extraction -- Starting " << endl;
    auto features = extractFeatures(src, superPixels, candidate_size);
    cout << "Feature Extraction -- Finished." << endl;

    /*--------------------------------- End Feature Extraction Phase ------------------------------*/

    return features;
}

void createCandidates (const string targets) {

    vector<String> fn = extractImagePath(targets);

    for (auto image : fn) {

        cout << endl << "---------------" << image << endl;

        auto features = preClassification(image);
        saveFeatures(features, "data", image, "features");

    }
}

int main(int argc, char*argv[]) {

    //Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
        return 0;
    } else {
        auto mode = string(argv[1]);

        if (mode == "-d" && argc > 2) {
            createCandidates(argv[2]);
        } else if (mode == "-i" && argc > 2) {

            /*--------------------------------- Classification Phase ------------------------------*/

            portable_mkdir("../results");

            auto features = preClassification(argv[2]);
            Mat labels((int) features.size(), 1, CV_32SC1);

            Classifier(features, 1000, "../data/svm_trained_model.yml", labels);

            for (int i = 0; i < features.size(); ++i) {
                if (labels.at<float>(0, i) == 1) {
                    imwrite("../results/Candidate_" + to_string(i) + ".bmp", features[i].candidate);
                }
            }

        } else if (mode == "-t") {

            /*--------------------------------- Training Phase ------------------------------*/

            Mat labels(0, 0, CV_32SC1);
            vector<Features> candidates;

            loadFromCSV("../data/features.csv", candidates, labels);

            Training(candidates, labels, 1000, "../data/svm_trained_model.yml");
        }
    }

    // Calculate and Print the execution time

    timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
    cout << "Times passed in seconds: " << timeElapsed << endl;

//    waitKey();
    return 1;
}


