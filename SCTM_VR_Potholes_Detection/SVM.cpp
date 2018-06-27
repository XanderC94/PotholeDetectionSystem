//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"

using namespace cv::ml;

namespace mysvm {
    Ptr<SVM> initSVM(const string model_path, int max_iter) {
        const double epsilon = exp(-6);

        Ptr<SVM> svm = SVM::create();

        try {
            svm = svm->load(model_path);
        } catch (Exception ex) {
            cout << "ERROR -- No saved model has been found... Training will start from scratch." << endl;
        }

        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::RBF);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));

        return svm;
    }

    void Classifier(const Mat &data, Mat &labels, const int max_iter, const string model_path) {

        Ptr<SVM> svm = initSVM(model_path, max_iter);

        svm->predict(data, labels);

        transpose(labels, labels);
        cout << labels << endl;
    }

    void Training(const Mat &data, const Mat &labels, const int max_iter, const string model_path) {

        cout << "FT size " << data.rows << "*" << data.cols << endl;

        printf("SVM Initialization\n");

        Ptr<SVM> svm = initSVM(model_path, max_iter);

        printf("Ready...\n");

        auto train_data = TrainData::create(data, ROW_SAMPLE, labels);

        svm->trainAuto(train_data, 3);

        printf("Finished.\n");

        svm->save(model_path);
    }
}