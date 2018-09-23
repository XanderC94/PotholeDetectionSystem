#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "phdetection/svm.hpp"
#include "phdetection/bayes.hpp"
#include "phdetection/io.hpp"
#include "phdetection/core.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace phd::io;

int numberFirstSPCandidatesFound = 0;

Configuration config;

const string config_folder = "../res/config";
const string data_folder = "../res/features";
const string results_folder = "../res/results";
const string svm_folder = "../res/svm";
const string nbayes_folder = "../res/bayes";

void showHelper() {

    cout << " Pothole Detection System Helper" << endl << endl;

    cout << "\t -d == generate candidates inside the given folder" << endl;
    cout << "\t\t Example: -d /x/y/z {-f}" << endl << endl;
    cout << "\t\t NOTE: The optional -f parameter will enable user-driven feedback for each candidate found."
         << endl << endl;

    cout << "\t -t == train an SVM, Bayes or Multi Classifier using the generated data" << endl;
    cout << "\t\t Example: -t -{svm, bayes, multi} /x/y/z/ model_name" << endl << endl;
    cout << "\t\t NOTE: no path or extension for the model name is required, " << endl
         << "\t\t it will be stored into ../res/svm/ or ../res/bayes/ inside the program folder." << endl << endl;

    cout << "\t -c == classify the given image or set of images (folder path)" << endl;
    cout << "\t\t Example: -c -{svm, bayes, multi} -{i, d} {/x/y/z, /x/y/z/image.something} model_name" << endl << endl;
    cout << "\t\t NOTE: no path or extension for the model name is required, " << endl
         << "\t\t it must be located inside the directories ../res/svm/ or ../res/bayes/ inside the program folder."
         << endl;
}

Mat go(const string &method, const string &model_name, const string &image, const Configuration &config) {

    cout << endl << "---------------" << image << endl;

    auto features = phd::getFeatures(image, config);

    cv::Mat labels;

    try {
        labels = phd::classify(method, svm_folder + "/" + model_name, nbayes_folder + "/" + model_name, features);
    } catch(phd::UndefinedMethod &ex)  {
        cerr << "ERROR: " << ex.what() << endl;
        exit(-1);
    }

    cout << "LABELS: " << labels << endl;

    for (int i = 0; i < features.size(); ++i) {

        string folder = results_folder + "/neg/";

        if (labels.at<int>(0, i) > 0 || labels.at<float>(0, i) > 0) {
            folder = results_folder + "/pos/";
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

void createCandidates(const string &targets, const bool feedback, const Configuration &config) {

    vector<string> fn = extractImagePath(targets);
    vector<string> names;
    vector<Features> features;

    for (auto image : fn) {

        cout << endl << "---------------" << image << endl;

        auto ft = phd::getFeatures(image, config);

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

    saveFeaturesJSON(features, data_folder, names, "features");

}

void classificationPhase(char*argv[], const Configuration &config){

    auto method = string(argv[2]);
    auto target_type = string(argv[3]);
    auto target = string(argv[4]);
    auto model_name = string(argv[5]);

    cout << "classify Method: " << method << endl;

    portable_mkdir(results_folder.data());
    portable_mkdir((results_folder + "/neg").data());
    portable_mkdir((results_folder + "/pos").data());

    /*--------------------------------- classify Phase ------------------------------*/
    int numberOfCandidatesFound = 0;
    if (target_type == "-d") { /// Whole folder

        vector<string> fn = extractImagePath(target);

        cout << "Number of image found in the directory: " << fn.size() << endl;
        for (const auto &image : fn) {
            numberOfCandidatesFound += go(method, model_name, image, config).cols;
        }
    } else if (target_type == "-i") { /// Single Image
        numberOfCandidatesFound += go(method, model_name, target, config).cols;
    }

    cout << "Candidates at first segmentation found: " << numberFirstSPCandidatesFound << endl;
    cout << "Candidates found: " << numberOfCandidatesFound << endl;
}

void trainingPhase(char*argv[]){

    auto method = string(argv[2]);
//    cout << "Training Method: " << method << endl;

    Mat labels(0, 0, CV_32SC1);
    vector<Features> candidates;

    loadFromJSON(data_folder + "/" + string(argv[3]), candidates, labels);

    /*--------------------------------- Training Phase ------------------------------*/
    bool trainBoth = (method == "-multi");

    if (trainBoth || method == "-svm") {

        /***************************** SVM CLASSIFIER ********************************/

        portable_mkdir(svm_folder.data());
        const auto model = svm_folder + "/" + string(argv[4]);
        phd::ml::svm::Training(candidates, labels, 1000, exp(-6), model);
    }

    if (trainBoth || method == "-bayes") {

        /***************************** Bayes CLASSIFIER ********************************/

        portable_mkdir(nbayes_folder.data());
        const auto std_model = nbayes_folder + "/" + string(argv[4]);
        phd::ml::bayes::Training(candidates, labels, std_model);
    }
}

int main(int argc, char *argv[]) {

    cout << phd::io::GetCurrentWorkingDir() << endl;

    // Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 2) {
        showHelper();
        return 0;
    } else {

        auto mode = string(argv[1]);

        config = loadProgramConfiguration(config_folder + "/config.json");

        printThresholds(config.primaryThresholds);
        printThresholds(config.secondaryThresholds);
        printOffsets(config.offsets);

        if (mode == "-d" && argc > 2) {
            createCandidates(argv[2], argc > 3 && string(argv[3]) == "-f", config);
        } else if (mode == "-c" && argc > 5) {
            classificationPhase(argv, config);
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


