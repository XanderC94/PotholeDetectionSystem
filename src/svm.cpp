//
// Created by Xander_C on 5/06/2018.
//

#include "phdetection/svm.hpp"
#include "phdetection/ml_utils.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace phd::ontologies;

namespace phd::ml::svm {

    void Classifier(const vector<Features> &features, Mat &labels, const string model_path) {

        cout << "Loading SVM from " << model_path << endl;

        Ptr<SVM> svm = SVM::load(model_path);
        //Ptr<SVM> svm = StatModel::load<SVM>(model_path);

        const Mat data = phd::ml::utils::ConvertFeaturesForSVM(features, svm->getVarCount());

        if (svm->isTrained()) {
            cout << "classify... ";
            svm->predict(data, labels);
            cout << "Finished." << endl;
            transpose(labels, labels);
            cout << "SVM Labels: " << labels << endl;
        } else {
            cerr << "SVM Classifier is not trained";
            exit(-1);
        }

        svm->clear();
    }

    void Training(const vector<Features> &features, const Mat &labels,
                  const int max_iter, const double epsilon, const string model_path) {

        const Mat dataFeatures = phd::ml::utils::ConvertFeaturesForSVM(features, 0);
        cout << "FT size " << dataFeatures.rows << "*" << dataFeatures.cols << endl;

        cout << "SVM Initialization..." << endl;

        Ptr<SVM> svm = SVM::create();

        try {
            svm = svm->load(model_path);
        } catch (Exception &ex) {
            cerr << "No saved model has been found... Training will start from scratch." << endl;
        }

        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::RBF);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));

        cout << "SVM Training...";

        auto train_data = TrainData::create(dataFeatures, ROW_SAMPLE, labels);

        svm->trainAuto(train_data,
                       10,
                       SVM::getDefaultGrid(SVM::C),
                       SVM::getDefaultGrid(SVM::GAMMA),
                       SVM::getDefaultGrid(SVM::P),
                       SVM::getDefaultGrid(SVM::NU),
                       SVM::getDefaultGrid(SVM::COEF),
                       SVM::getDefaultGrid(SVM::DEGREE),
                       true);

        cout << "Finished." << endl;

        svm->save(model_path);

        cout << "Model Saved." << endl;

    }
}