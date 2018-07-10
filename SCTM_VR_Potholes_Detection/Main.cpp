#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Segmentation.h"
#include "FeaturesExtraction.h"
#include "SVM.h"
#include "Bayes.h"
#include "Utils.h"
#include "MLUtils.h"

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace mlutils;

int numberFirstSPCandidatesFound = 0;

Configuration config;

void showHelper() {

    cout << " Pothole Detection System Helper" << endl << endl;

    cout << "\t -d == generate candidates inside the given folder" << endl;
    cout << "\t\t Example: -d /x/y/z {-f}" << endl << endl;
    cout << "\t\t NOTE: The optional -f parameter will enable user-driven feedback for each candidate found."
         << endl << endl;

    cout << "\t -t == train an SVM, Bayes or Multi Classifier using the generated data" << endl;
    cout << "\t\t Example: -t -{svm, bayes, multi} /x/y/z/ model_name" << endl << endl;
    cout << "\t\t NOTE: no path or extension for the model name is required, " << endl
         << "\t\t it will be stored into ../svm/ or ../bayes/ inside the program folder." << endl << endl;

    cout << "\t -c == classify the given image or set of images (folder path)" << endl;
    cout << "\t\t Example: -c -{svm, bayes, multi} -{i, d} {/x/y/z, /x/y/z/image.something} model_name" << endl << endl;
    cout << "\t\t NOTE: no path or extension for the model name is required, " << endl
         << "\t\t it must be located inside the directories ../svm/ or ../bayes/ inside the program folder."
         << endl;
}

vector<Features> getFeatures(const string &target) {

    auto candidateSuperPixels = vector<SuperPixel>();

    auto candidate_size = Size(128, 128);

    /*---------------------------------Load image------------------------*/

    Mat src = imread(target, IMREAD_COLOR), tmp;

    /*
    *	The src image will be:
    *	1. Resized
    *	2. Cropped under the Horizon Line
    *   3. Segmented with superpixeling
    *   4. Superpixels in the RoI is selected using Analytic Rect Function
    *	(in Candidates will be placed the set on candidateSuperPixels that survived segmentation)
    */

    int superPixelEdge = 32;

    /*--------------------------------- Pre-Processing Phase ------------------------------*/

    cout << "Pre-Processing... ";
    preprocessing(src, src, config.offsets.horizon);
    cout << "Finished." << endl;

    /*--------------------------------- End Pre-Processing Phase ------------------------------*/
//    showElaborationStatusToTheUser("Preprocessing Result", src);

    /*--------------------------------- First Segmentation Phase ------------------------------*/

    cout << "Segmentation... " << endl;
    // NB: Init once, use many times!
    auto superPixeler = initSuperPixelingLSC(src, superPixelEdge);
    extractRegionsOfInterest(superPixeler, src, candidateSuperPixels,
                             superPixelEdge, config.primaryThresholds, config.offsets);
    cout << "Finished." << endl;
    cout << "Found " << candidateSuperPixels.size() << " candidates." << endl;
    numberFirstSPCandidatesFound += candidateSuperPixels.size();
    /*--------------------------------- End First Segmentation Phase ------------------------------*/

    /*--------------------------------- Feature Extraction Phase ------------------------------*/

    cout << "Feature Extraction -- Started. " << endl;
    auto features = extractFeatures(src, candidateSuperPixels, candidate_size,
                                    config.offsets, config.secondaryThresholds);
    cout << "Feature Extraction -- Finished." << endl;

    /*--------------------------------- End Feature Extraction Phase ------------------------------*/
//
//    showElaborationStatusToTheUser(features);

    return features;
}

Mat Classification(const string &method, const string &model_name, const vector<Features> &features) {

    Mat std_labels(static_cast<int>( features.size()), 1, CV_32SC1);

    if (!features.empty()) {

        if (method == "-svm") {

            /***************************** SVM CLASSIFIER ********************************/
            const auto svm_model = "../svm/" + model_name;
            mySVM::Classifier(features, std_labels, svm_model);

        } else if (method == "-bayes") {

            /***************************** Bayes CLASSIFIER ********************************/
            const auto std_model = "../bayes/" + model_name;
            myBayes::Classifier(features, std_labels, std_model);

        } else if (method == "-multi") {
            /***************************** MULTI CLASSIFIER ********************************/
            Mat svm_labels(static_cast<int>(features.size()), 1, CV_32SC1);
            const auto svm_model = "../svm/" + model_name;
            mySVM::Classifier(features, svm_labels, svm_model);

            Mat bayes_labels(static_cast<int>(features.size()), 1, CV_32SC1);
            const auto std_model = "../bayes/" + model_name;
            myBayes::Classifier(features, bayes_labels, std_model);

            std_labels = mergeMultiClassifierResults(svm_labels, bayes_labels);
        } else {
            cerr << "Undefined method " << method << endl;

            showHelper();

            exit(-1);
        }

    } else return Mat();

    return std_labels;
}

Mat go(const string &method, const string &model_name, const string &image) {

    cout << endl << "---------------" << image << endl;

    auto features = getFeatures(image);

    auto labels = Classification(method, model_name, features);

    cout << "LABELS: " << labels << endl;

    for (int i = 0; i < features.size(); ++i) {

        string folder = "../results/neg/";

        if (labels.at<int>(0, i) > 0 || labels.at<float>(0, i) > 0) {
            folder = "../results/pos/";
        }

        imwrite(
                folder +
                extractFileName(set_format(image, "", false), "/") +
                "_L" + to_string(features[i].label) +
                "_" + to_string(features[i].id) +
                ".bmp",
                features[i].candidate
        );
    }

    return labels;
}

/*
 * This function show the candidate extracted and ask if is pothole (Y) or not (N)
 * */
int askUserSupervisionBinaryClasses(const Features &candidateFeatures, const int scaleFactor = 5) {

    Mat visual_image;
    resize(candidateFeatures.candidate,
           visual_image,
           Size(candidateFeatures.candidate.cols * scaleFactor,
                candidateFeatures.candidate.rows * scaleFactor));

    imshow("Is pothole?", visual_image);
    waitKey(1);
    cout << "This candidate is a pothole? Y (for response yes) N (for no)" << endl;
    int result = -1;
    char response;
    bool isResponseCorrect = false;
    while(!isResponseCorrect) {
        cin >> response;
        if (response == 'P' || response == 'y') {
            isResponseCorrect = true;
            result = 1;
        } else if (response == 'N' || response == 'n') {
            result = -1;
            isResponseCorrect = true;
        } else {
            cout << "Wrong response. Type (Y) for yes or (N) for no" << endl;
        }
    }
    return result;
}

int askUserSupervisionMultiClasses(const Features &candidateFeatures, const int scaleFactor = 5) {

    Mat visual_image;
    resize(candidateFeatures.candidate,
           visual_image,
           Size(candidateFeatures.candidate.cols * scaleFactor, candidateFeatures.candidate.rows * scaleFactor));

    imshow("Is pothole?", visual_image);
    waitKey(1);
    cout << "This candidate wich type of pothole is? "
            "P (for response Pothole), C (for street crack), O (for out of road), S (for street/car/sidewalk)"
         << endl;
    int result = -1;
    char response;
    bool isResponseCorrect = false;
    while (!isResponseCorrect) {
        cin >> response;
        if (response == 'P' || response == 'p') {
            isResponseCorrect = true;
            result = ClassificationClasses::pothole;
        } else if (response == 'C' || response == 'c') {
            result = ClassificationClasses::asphaltCrack;
            isResponseCorrect = true;
        } else if (response == 'O' || response == 'o') {
            result = ClassificationClasses::outOfRoad;
            isResponseCorrect = true;
        } else if (response == 'S' || response == 's') {
            result = ClassificationClasses::streetSideWalkOrCar;
            isResponseCorrect = true;
        } else {
            cout
                    << "Wrong response. Type P (for response Pothole), C (for street crack), O (for out of road), S (for street/car/sidewalk)"
                    << endl;
        }
    }
    return result;
}

void createCandidates(const string &targets, const bool feedback) {

    vector<String> fn = extractImagePath(targets);
    vector<string> names;
    vector<Features> features;

    for (auto image : fn) {

        cout << endl << "---------------" << image << endl;

        auto ft = getFeatures(image);

        for (auto f : ft) {
            names.push_back(image);
            if (feedback) {
                //f._class = askUserSupervisionBinaryClasses(f);
                f._class = askUserSupervisionMultiClasses(f);
                cout << " Image " << image << " L" << f.label
                     << " labeled as " << (f._class == 1 ? "pothole." : "not pothole.")
                     << endl;
            }

            features.push_back(f);
        }
    }

    saveFeaturesJSON(features, "data", names, "features");

}

void classificationPhase(char*argv[]){
    auto method = string(argv[2]);
    auto target_type = string(argv[3]);
    auto target = string(argv[4]);
    auto model_name = string(argv[5]);

    cout << "Classification Method: " << method << endl;

    portable_mkdir("../results");
    portable_mkdir("../results/neg");
    portable_mkdir("../results/pos");

    /*--------------------------------- Classification Phase ------------------------------*/
    int numberOfCandidatesFound = 0;
    if (target_type == "-d") { /// Whole folder

        vector<String> fn = extractImagePath(target);

        cout << "Number of image found in the directory: " << fn.size() << endl;
        for (const auto &image : fn) {
            numberOfCandidatesFound += go(method, model_name, image).cols;
        }
    } else if (target_type == "-i") { /// Single Image
        numberOfCandidatesFound += go(method, model_name, target).cols;
    }

    cout << "Candidates at first segmentation found: " << numberFirstSPCandidatesFound << endl;
    cout << "Candidates found: " << numberOfCandidatesFound << endl;
}

void trainingPhase(char*argv[]){

    auto method = string(argv[2]);
    cout << "Training Method: " << method << endl;

    Mat labels(0, 0, CV_32SC1);
    vector<Features> candidates;

    loadFromJSON("../data/" + string(argv[3]), candidates, labels);

    /*--------------------------------- Training Phase ------------------------------*/
    bool trainBoth = (method == "-multi");

    if (trainBoth || method == "-svm") {

        /***************************** SVM CLASSIFIER ********************************/

        portable_mkdir("../svm/");
        const auto model = "../svm/" + string(argv[4]);
        mySVM::Training(candidates, labels, 1000, exp(-6), model);
    }

    if (trainBoth || method == "-bayes") {

        /***************************** Bayes CLASSIFIER ********************************/

        portable_mkdir("../bayes/");
        const auto std_model = "../bayes/" + string(argv[4]);
        myBayes::Training(candidates, labels, std_model);
    }
}

int main(int argc, char *argv[]) {

    // Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
        showHelper();
        return 0;
    } else {

        auto mode = string(argv[1]);

        config = loadProgramConfiguration("../config/config.json");

        printThresholds(config.primaryThresholds);
        printThresholds(config.secondaryThresholds);
        printOffsets(config.offsets);

        if (mode == "-d" && argc > 2) {
            createCandidates(argv[2], argc > 3 && string(argv[3]) == "-f");
        } else if (mode == "-c" && argc > 5) {
            classificationPhase(argv);
        } else if (mode == "-t" && argc >= 4) {
            trainingPhase(argv);
        } else {
            showHelper();
        }
    }

    // Calculate and Print the execution time

    timeElapsed = (static_cast<double>( getTickCount()) - timeElapsed) / getTickFrequency();
    cout << "Times passed in seconds: " << timeElapsed << endl;

    return 1;
}


