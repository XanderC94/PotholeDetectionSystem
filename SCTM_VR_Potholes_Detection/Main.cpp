#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/plot.hpp>
#include "Segmentation.h"
#include "FeaturesExtraction.h"
#include "SVM.h"
#include "Utils.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

vector<Features> preClassification(const string &target) {

    auto candidateSuperPixels = vector<SuperPixel>();

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
    *	(in Candidates will be placed the set on candidateSuperPixels that survived segmentation)
    */

    RoadOffsets offsets = {
            .Horizon_Offset = 0.65,
            .SLine_X_Right_Offset = 0.0,
            .SLine_X_Left_Offset = 0.25,
            .SLine_Y_Right_Offset = 0.9,
            .SLine_Y_Left_Offset = 0.9,
            .SLine_Right_Escape_Offset = 0.4,
            .SLine_Left_Escape_Offset = 0.4
    };

    ExtractionThresholds thresholds = {
            .Density_Threshold = 0.55, // OK, do not change
            .Variance_Threshold = 0.3,
            .Gauss_RoadThreshold = 0.60,
            .colourRatioThresholdMin= 1.15, // 1.25 is better if we aim to properly detect nearest holes to the vehicle,
            // but will probably exclude far away holes and those holes near cars (see image of test n_89)
            .colourRatioThresholdMax = 2.5
    };

    int superPixelEdge = 32;

    printThresholds(thresholds);
    printOffsets(offsets);

    /*--------------------------------- Pre-Processing Phase ------------------------------*/

    cout << "Pre-Processing... ";
    preprocessing(src, src, offsets.Horizon_Offset);
    cout << "Finished." << endl;

    /*--------------------------------- End Pre-Processing Phase ------------------------------*/

    /*--------------------------------- First Segmentation Phase ------------------------------*/

    cout << "Segmentation... " << endl;
    // NB: Init once, use many times!
    auto superPixeler = initSuperPixelingLSC(src, superPixelEdge);
    extractRegionsOfInterest(superPixeler, src, candidateSuperPixels,
                             superPixelEdge, thresholds, offsets);
    cout << "Finished." << endl;

    cout << "Found " << candidateSuperPixels.size() << " candidates." << endl;

    /*--------------------------------- End First Segmentation Phase ------------------------------*/

    /*--------------------------------- Feature Extraction Phase ------------------------------*/

    cout << "Feature Extraction -- Started. " << endl;
    auto features = extractFeatures(src, candidateSuperPixels, candidate_size, offsets, thresholds);
    cout << "Feature Extraction -- Finished." << endl;

    /*--------------------------------- End Feature Extraction Phase ------------------------------*/

    return features;
}

void createCandidates (const string targets) {

    vector<String> fn = extractImagePath(targets);
    vector<string> names;
    vector<Features> features;

    for (auto image : fn) {

        cout << endl << "---------------" << image << endl;

        auto ft = preClassification(image);

        for (auto f : ft) {
            names.push_back(image);
            features.push_back(f);
        }
    }

//    saveFeaturesCSV(features, "data", names, "features");
    saveFeaturesJSON(features, "data", names, "features");

}

int main(int argc, char*argv[]) {

    // Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
        return 0;
    } else {
        auto mode = string(argv[1]);

        if (mode == "-d" && argc > 2) {
            createCandidates(string(argv[2]));
        } else if (mode == "-i" && argc > 3) {

            /*--------------------------------- Classification Phase ------------------------------*/

            portable_mkdir("../results");

            auto features = preClassification(string(argv[2]));
            Mat labels((int) features.size(), 1, CV_32SC1);

            if (features.size() > 0) {
                Classifier(features, 1000, "../svm/" + string(argv[3]), labels);

                for (int i = 0; i < features.size(); ++i) {
//                if (labels.at<float>(0, i) == 1) {
                    imwrite("../results/Candidate_" + to_string(i) + (labels.at<float>(0, i) == 1 ? "_Pos" : "_Neg") +
                            ".bmp", features[i].candidate);
//                }
                }
            }

        } else if (mode == "-t" && argc > 3) {

            /*--------------------------------- Training Phase ------------------------------*/

            Mat labels(0, 0, CV_32SC1);
            vector<Features> candidates;

//            loadFromCSV("../data/" + string(argv[2]), candidates, labels);
            loadFromJSON("../data/" + string(argv[2]), candidates, labels);

            portable_mkdir("../svm/");

            Training(candidates, labels, 1000, "../svm/" + string(argv[3]));
        }
    }

    // Calculate and Print the execution time

    timeElapsed = ((double) getTickCount() - timeElapsed) / getTickFrequency();
    cout << "Times passed in seconds: " << timeElapsed << endl;

    //waitKey();
    return 1;
}


