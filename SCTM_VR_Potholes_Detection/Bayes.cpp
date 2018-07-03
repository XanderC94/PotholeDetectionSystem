//
// Created by Xander_C on 27/06/2018.
//

#include "Bayes.h"
#include "MLUtils.h"

namespace myBayes {

    void Classifier(const vector<Features> &features, Mat &labels, const string model_path) {
        auto bayes = cv::ml::NormalBayesClassifier::load(model_path);

        const Mat dataFeatures = mlutils::ConvertFeaturesForSVM(features, bayes->getVarCount());

        if (bayes->isTrained()) {
            cout << "Classification... ";
            bayes->predict(dataFeatures, labels);
            cout << "Finished." << endl;
            transpose(labels, labels);
        } else {
            cerr << "Bayes Classifier is not trained";
            exit(-1);
        }
    }

    int handleError(int status, const char *func_name,
                    const char *err_msg, const char *file_name,
                    int line, void *userdata) {
        //Do nothing -- will suppress console output
        return 0;   //Return value is not used
    }

    void Training(const vector<Features> &features, const Mat &labels, const string model_path) {

        cv::redirectError(handleError);
        const Mat dataFeatures = mlutils::ConvertFeaturesForSVM(features, 0);

        auto bayes = ml::NormalBayesClassifier::create();

        try {

            bayes = bayes->load(model_path);
        } catch (const exception ex) {
            cerr << "No saved model has been found... Training will start from scratch." << endl;
        }

        cv::redirectError(nullptr);

        auto train_data = ml::TrainData::create(dataFeatures, ml::ROW_SAMPLE, labels);

        cout << "Training... ";

        bayes->train(train_data);

        cout << "Finished." << endl;

        Mat outputs(dataFeatures.rows, 1, CV_32S);

        auto error = bayes->calcError(train_data, false, outputs);

        cout << "Calculated Classification Error: " << error << endl;

        bayes->save(model_path);

        bayes->clear();
    }

}