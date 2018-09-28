#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libgen.h>

#include "phdetection/svm.hpp"
#include "phdetection/bayes.hpp"
#include "phdetection/io.hpp"
#include "phdetection/core.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace phd::io;
using namespace phd::ontologies;

int numberFirstSPCandidatesFound = 0;

Configuration config;

string config_folder = "/res/config";
string data_folder = "/res/features";
string results_folder = "/res/results";
string svm_folder = "/res/svm";
string nbayes_folder = "/res/bayes";

typedef struct ARGS {
    string mode;
    string target_type;
    string target;
    bool user_feedback;
    string method;
    string bayes_model;
    string svm_model;
} ARGS;

void showHelper() {

    cout << " Pothole Detection System Helper" << endl << endl;

    cout << " -g == generate candidates inside the given folder" << endl << endl;
    cout << "Example: -g -{i, d} {/path/to/image.abc, /path/to/images/} {-uf}" << endl << endl;
    cout << "NOTE: The optional -uf parameter will enable user-driven feedback for each candidate found." << endl;
    cout << "NOTE: This will create a features.json features file under ../res/features/ ."
         << endl << endl;

    cout << " -t == train an SVM, Bayes or Multi Classifier using the generated data" << endl << endl;
    cout << "Example: -t -ft /path/to/features.json "
            << "-{svm, bayes, multi} -b /path/to/bayes/model.yml -s /path/to/svm/model.yml" << endl << endl;

    cout << " -c == classify the given image or set of images (folder path)" << endl << endl;
    cout << "Example: -c -{i, d} {/path/to/image.abc, /path/to/images/} "
         << "-{svm, bayes, multi} -b /path/to/bayes/model.yml -s /path/to/svm/model.yml" << endl << endl;
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

void _create_candidates_stub(
        const std::string image,
        const bool user_feedback,
        const Configuration &config,
        vector<Features> &features,
        vector<string> &names) {

    auto ft = phd::getFeatures(image, config);

    for (auto f : ft) {
        names.push_back(image);
        if (user_feedback) {
            //f._class = askUserSupervisionBinaryClasses(f);
            f._class = askUserSupervisionMultiClasses(f);
            cout << " Image " << image << " L" << f.label
                 << " labeled as " << (f._class == 1 ? "pothole." : "not pothole.")
                 << endl;
        }

        features.push_back(f);
    }
}

void createCandidates(const ARGS args, const Configuration &config) {

    vector<string> names;
    vector<Features> features;

    if (args.target_type == "-d") {
        vector<string> fn = extractImagePath(args.target);

        for (auto image : fn) {

            cout << endl << "---------------" << image << endl;

            _create_candidates_stub(image, args.user_feedback, config, features, names);
        }
    } else if (args.target_type == "-i") {

        _create_candidates_stub(args.target, args.user_feedback, config, features, names);
    }

    saveFeaturesJSON(features, names, data_folder + "/features.json");

}

Mat _classifier_stub(const string &method, const string &bayes_model, const string &svm_model, const string &image,
                     const Configuration &config) {

    cout << endl << "---------------" << image << endl;

    auto features = phd::getFeatures(image, config);

    cv::Mat labels;

    try {
        labels = phd::classify(method, svm_model, bayes_model, features);
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
                getName(set_format(image, "", false)) +
                "_L" + to_string(features[i].label) +
                "_" + to_string(features[i].id) +
                ".bmp",
                features[i].candidate
        );
    }

    return labels;
}

void classificationPhase(const ARGS args, const Configuration &config){

    cout << "classify Method: " << args.method << endl;

    portable_mkdir(results_folder.data());
    portable_mkdir((results_folder + "/neg").data());
    portable_mkdir((results_folder + "/pos").data());

    /*--------------------------------- classify Phase ------------------------------*/
    int numberOfCandidatesFound = 0;
    if (args.target_type == "-d") { /// Whole folder

        vector<string> fn = extractImagePath(args.target);

        cout << "Number of image found in the directory: " << fn.size() << endl;
        for (const auto &image : fn) {
            numberOfCandidatesFound += _classifier_stub(args.method, args.bayes_model, args.svm_model, image, config).cols;
        }
    } else if (args.target_type == "-i") { /// Single Image
        numberOfCandidatesFound += _classifier_stub(args.method, args.bayes_model, args.svm_model, args.target, config).cols;
    } else {
        cerr << "Unknown target type " << args.target_type << endl;
        exit(-1);
    }

    cout << "Candidates at first segmentation found: " << numberFirstSPCandidatesFound << endl;
    cout << "Candidates found: " << numberOfCandidatesFound << endl;
}

void trainingPhase(const ARGS args, const Configuration config){

    cout << "Training Method: " << args.method << endl;

    Mat labels(0, 0, CV_32SC1);
    vector<Features> candidates;

    loadFromJSON(args.target, candidates, labels);

    /*--------------------------------- Training Phase ------------------------------*/
    bool trainBoth = (args.method == "-multi");

    if (trainBoth || args.method == "-svm") {

        /***************************** SVM CLASSIFIER ********************************/

        portable_mkdir(svm_folder.data());
        phd::ml::svm::Training(candidates, labels, 1000, exp(-6), args.svm_model);
    }

    if (trainBoth || args.method == "-bayes") {

        /***************************** Bayes CLASSIFIER ********************************/

        portable_mkdir(nbayes_folder.data());
        phd::ml::bayes::Training(candidates, labels, args.bayes_model);
    }
}


ARGS parseCommandLineArguments(int argc, char*argv[]) {

    auto mode = string(argv[1]);
    auto target_type = string(argv[2]);
    auto targets = string(argv[3]);
    auto method = string();
    auto bayes_model = string();
    auto svm_model = string();
    auto user_feedback = false;

    if (mode == "-g") {
        user_feedback = argc > 4 && string(argv[4]) == "-uf";
    } else if (mode == "-c" || mode == "-t"){
        method = string(argv[4]);

        if (method == "-bayes") {
            int idx = string(argv[5]) == "-b" ? 6 : 5;
            bayes_model = string(argv[idx]);
        } else if(method == "-svm") {
            int idx = string(argv[5]) == "-s" ? 6 : 5;
            svm_model = string(argv[6]);
        } else if (method == "-multi" && argc > 7 && (
                (string(argv[5]) == "-b" && string(argv[7]) == "-s") ||
                (string(argv[7]) == "-b" && string(argv[5]) == "-s"))) {

            if (string(argv[5]) == "-b") {
                bayes_model = string(argv[6]);
                svm_model = string(argv[8]);
            } else {
                bayes_model = string(argv[8]);
                svm_model = string(argv[6]);
            }
        } else {
            showHelper();
            exit(-1);
        }

    } else {
        showHelper();
        exit(-1);
    }

    cout << "FOUND: " << endl
        << "Mode:" << mode << endl
        << "TT:" << target_type << endl
        << "Target:" << targets << endl
        << "feedback:" << user_feedback << endl
        << "Method:" << method << endl
        << "SVM:" << bayes_model << endl
        << "BAYES:" << svm_model << endl << endl;


    return ARGS {mode, target_type, targets, user_feedback, method, bayes_model, svm_model};
}


int main(int argc, char *argv[]) {

//    cout << phd::io::GetCurrentWorkingDir() << endl;

    const string root = phd::io::getParentDirectory(string(dirname(argv[0])));

    config_folder = root + config_folder;
    data_folder = root + data_folder;
    results_folder = root + results_folder;
    svm_folder = root + svm_folder;
    nbayes_folder = root + nbayes_folder;

//    cout << config_folder << endl;
//    cout << data_folder << endl;
//    cout << results_folder << endl;
//    cout << svm_folder << endl;
//    cout << nbayes_folder << endl;

    // Save cpu tick count at the program start
    double timeElapsed = (double) getTickCount();

    if (argc < 3) {
        showHelper();
        return 0;
    } else {

        config = loadProgramConfiguration(config_folder + "/config.json");

        printThresholds(config.primaryThresholds);
        printThresholds(config.secondaryThresholds);
        printOffsets(config.offsets);

        const auto args = parseCommandLineArguments(argc, argv);



        if (args.mode == "-g" && argc > 2) {
            createCandidates(args, config);
        } else if (args.mode == "-c" && argc > 6) {
            classificationPhase(args, config);
        } else if (args.mode == "-t" && argc >= 5) {
            trainingPhase(args, config);
        } else {
            showHelper();
        }
    }

    // Calculate and Print the execution time

    timeElapsed = (static_cast<double>( getTickCount()) - timeElapsed) / getTickFrequency();
    cout << "Times passed in seconds: " << timeElapsed << endl;

    return 1;
}


