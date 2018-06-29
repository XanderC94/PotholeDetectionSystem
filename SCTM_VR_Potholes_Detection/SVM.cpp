//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"
#include "MLUtils.h"

using namespace cv::ml;

namespace mySVM {

    void Classifier(const vector<Features> &features, Mat &labels, const int max_iter, const string model_path) {


        Ptr<SVM> svm = ml::SVM::load(model_path);
        //Ptr<SVM> svm = StatModel::load<SVM>(model_path);

        const Mat data = mlutils::ConvertFeaturesForSVM(features, svm->getVarCount());

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
    Training(const vector<Features> &features, const Mat &labels, const int max_iter, const double epsilon, const string model_path) {

        const Mat dataFeatures = mlutils::ConvertFeaturesForSVM(features, 0);
        cout << "FT size " << dataFeatures.rows << "*" << dataFeatures.cols << endl;

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

        auto train_data = TrainData::create(dataFeatures, ROW_SAMPLE, labels);

        svm->trainAuto(train_data,
                       3,
                       SVM::getDefaultGrid(SVM::C),
                       SVM::getDefaultGrid(SVM::GAMMA),
                       SVM::getDefaultGrid(SVM::P),
                       SVM::getDefaultGrid(SVM::NU),
                       SVM::getDefaultGrid(SVM::COEF),
                       SVM::getDefaultGrid(SVM::DEGREE),
                       true);

        printf("Finished.\n");

        svm->save(model_path);

        printf("Saved model.\n");
    }
}