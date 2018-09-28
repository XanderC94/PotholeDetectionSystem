//
// Created by Xander on 19/9/2018.
//


#include "../include/phdetection/core.hpp"

#include "phdetection/ml_utils.hpp"
#include "phdetection/svm.hpp"
#include "phdetection/bayes.hpp"
#include "phdetection/segmentation.hpp"
#include "phdetection/features_extraction.hpp"
#include "phdetection/svm.hpp"
#include "phdetection/bayes.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace phd::ml::utils;
using namespace phd::ontologies;

namespace phd {

    vector<Features> getFeatures(Mat &target, const phd::io::Configuration &config) {

        auto candidateSuperPixels = vector<SuperPixel>();

        auto candidate_size = Size(128, 128);

        /*
        *	The target image will be:
        *	1. Resized
        *	2. Cropped under the Horizon Line
        *   3. Segmented with superpixeling
        *   4. Superpixels in the RoI is selected using Analytic Rect Function
        *	(in Candidates will be placed the set on candidateSuperPixels that survived segmentation)
        */

        int superPixelEdge = 32;

        /*--------------------------------- Pre-Processing Phase ------------------------------*/

        cout << "Pre-Processing... ";
        phd::segmentation::preprocessing(target, target, config.offsets.horizon);
        cout << "Finished." << endl;

        /*--------------------------------- End Pre-Processing Phase ------------------------------*/
//    showElaborationStatusToTheUser("Preprocessing Result", target);

        /*--------------------------------- First Segmentation Phase ------------------------------*/

        cout << "Segmentation... " << endl;
        // NB: Init once, use many times!
        auto superPixeler = phd::superpixeling::initSuperPixelingLSC(target, superPixelEdge);
        phd::segmentation::extractRegionsOfInterest(superPixeler, target, candidateSuperPixels,
                                                    superPixelEdge, config.primaryThresholds, config.offsets);
        cout << "Finished." << endl;
        cout << "Found " << candidateSuperPixels.size() << " candidates." << endl;
//    numberFirstSPCandidatesFound += candidateSuperPixels.size();
        /*--------------------------------- End First Segmentation Phase ------------------------------*/

        /*--------------------------------- Feature Extraction Phase ------------------------------*/

        cout << "Feature Extraction -- Started. " << endl;
        auto features = phd::features::extractFeatures(target, candidateSuperPixels, candidate_size,
                                                       config.offsets, config.secondaryThresholds);
        cout << "Feature Extraction -- Finished." << endl;

        /*--------------------------------- End Feature Extraction Phase ------------------------------*/
//
//    showElaborationStatusToTheUser(features);

        return features;
    }

    vector<Features> getFeatures(const string &target, const phd::io::Configuration &config) {
        /*---------------------------------Load image------------------------*/

        Mat src = imread(target, IMREAD_COLOR);

        return getFeatures(src, config);
    }

    Mat classify(const string &method, const string &svm_model,
                 const string &bayes_model, const vector<Features> &features) {

        Mat std_labels(static_cast<int>(features.size()), 1, CV_32SC1);

        if (!features.empty()) {

            if (method == "-svm") {

                /***************************** SVM CLASSIFIER ********************************/
                phd::ml::svm::Classifier(features, std_labels, svm_model);

            } else if (method == "-bayes") {

                /***************************** Bayes CLASSIFIER ********************************/
                phd::ml::bayes::Classifier(features, std_labels, bayes_model);

            } else if (method == "-multi") {
                /***************************** MULTI CLASSIFIER ********************************/
                Mat svm_labels(static_cast<int>(features.size()), 1, CV_32SC1);
                phd::ml::svm::Classifier(features, svm_labels, svm_model);

                Mat bayes_labels(static_cast<int>(features.size()), 1, CV_32SC1);
                phd::ml::bayes::Classifier(features, bayes_labels, bayes_model);

                std_labels = mergeMultiClassifierResults(svm_labels, bayes_labels);
            } else {

                throw UndefinedMethod("Undefined classification method " + method);
            }

        } else return Mat();

        return std_labels;
    }
}
