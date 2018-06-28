//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"

using namespace cv::ml;

namespace mySVM {

    void Classifier(const Mat &data, Mat &labels, const int max_iter, const string model_path) {

        Ptr<SVM> svm = ml::SVM::load(model_path);

        if (svm->isTrained()) {
            cout << "Classification... ";
            svm->predict(data, labels);
            cout << "Finished." << endl;
            transpose(labels, labels);
            cout << labels << endl;
        } else {
            cerr << "SVM Classifier is not trained";
            exit(-1);
        }
    }

    void
    Training(const Mat &data, const Mat &labels, const int max_iter, const double epsilon, const string model_path) {

        cout << "FT size " << data.rows << "*" << data.cols << endl;

        printf("SVM Initialization\n");

        Ptr<SVM> svm = SVM::create();

        try {
            svm = svm->load(model_path);
        } catch (Exception ex) {
            cerr << "No saved model has been found... Training will start from scratch." << endl;
        }

        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::RBF);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));

        printf("Ready...\n");

        auto train_data = TrainData::create(data, ROW_SAMPLE, labels);

        svm->trainAuto(train_data, 3);

        printf("Finished.\n");

        svm->save(model_path);
    }
}