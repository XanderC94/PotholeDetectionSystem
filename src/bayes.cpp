//
// Created by Xander_C on 27/06/2018.
//

#include "phdetection/bayes.hpp"
#include "phdetection/ml_utils.hpp"

using namespace std;
using namespace cv;
using namespace std;
using namespace phd::ontologies;

namespace phd::ml::bayes {

    void Classifier(const vector<Features> &features, Mat &labels, const string model_path) {

        cout << "Loading nBayes from " << model_path << endl;

        auto bayes = cv::ml::NormalBayesClassifier::load(model_path);

        const Mat dataFeatures = phd::ml::utils::ConvertFeaturesForBayes(features);

        if (bayes->isTrained()) {
            cout << "classify... ";
            bayes->predict(dataFeatures, labels);
            cout << "Finished." << endl;
            transpose(labels, labels);
            cout << "BAYES Labels: " << labels << endl;
        } else {
            cerr << "Bayes Classifier is not trained";
            exit(-1);
        }

        bayes->clear();
    }

    int handleError(int status, const char *func_name,
                    const char *err_msg, const char *file_name,
                    int line, void *userdata) {
        //Do nothing -- will suppress console output
        return 0;   //Return value is not used
    }

    void Training(const vector<Features> &features, const Mat &labels, const string model_path) {

        cv::redirectError(handleError);
        const Mat dataFeatures = phd::ml::utils::ConvertFeaturesForBayes(features);

        auto bayes = cv::ml::NormalBayesClassifier::create();

        try {
            bayes = bayes->load(model_path);
        } catch (const exception ex) {
            cerr << "No saved model has been found... Training will start from scratch." << endl;
        }

        cv::redirectError(nullptr);

        auto train_data = cv::ml::TrainData::create(dataFeatures, cv::ml::ROW_SAMPLE, labels);

        cout << "Bayes Training...";

        bayes->train(train_data);

        cout << "Finished." << endl;

        Mat outputs(dataFeatures.rows, 1, CV_32S);

        auto error = bayes->calcError(train_data, false, outputs);

        cout << "Calculated classify Error: " << error << endl;

        bayes->save(model_path);

        cout << "Model Saved." << endl;

        bayes->clear();
    }

}