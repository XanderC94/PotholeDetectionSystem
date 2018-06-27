//
// Created by Xander_C on 27/06/2018.
//

#include "Bayes.h"

namespace myBayes {

    void Classifier(const Mat &features, Mat &labels, const string model_path) {
        auto bayes = cv::ml::NormalBayesClassifier::load(model_path);

        if (bayes->isTrained()) {
            bayes->predict(features, labels);
            transpose(labels, labels);
            cout << labels << endl;
        } else {
            cerr << "Bayes Classifier is not trained";
            exit(-1);
        }
    }

    void Training(const Mat &features, const Mat &labels, const string model_path) {

        auto bayes = ml::NormalBayesClassifier::create();

        try {
            bayes = bayes->load(model_path);
        } catch (const exception ex) {
            cerr << "No saved model has been found... Training will start from scratch." << endl;
        }

        auto train_data = ml::TrainData::create(features, ml::ROW_SAMPLE, labels);

        bayes->train(train_data);

        bayes->save(model_path);
    }

}