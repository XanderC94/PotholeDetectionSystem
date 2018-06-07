//
// Created by Xander_C on 5/06/2018.
//

#include <thread>
#include "SVM.h"


Ptr<SVM> initSVM(String model_path, int max_iter) {
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

Mat ConvertFeatures(vector<Features> &features) {

    Mat data((int) features.size(), 5, CV_32F);

    for (int i = 0; i < features.size(); i++) {
        data.at<float>(i, 0) = features[i].averageGrayValue;
        data.at<float>(i, 1) = features[i].contrast;
        data.at<float>(i, 2) = features[i].skewness;
        data.at<float>(i, 3) = features[i].energy;
        data.at<float>(i, 4) = features[i].entropy;
    }

    return data;
}

Mat Classifier(vector<Features> &features, int max_iter, String model_path){

    Mat labels_mat((int) features.size(), 1, CV_32SC1);
    Mat data_mat = ConvertFeatures(features);

    Ptr<SVM> svm = initSVM(model_path, max_iter);

    auto flag = svm->predict(data_mat, labels_mat);

    printf("FLAG: %2f.5", flag);

    return labels_mat;
}

void Training(vector<Features> &features, vector<int> &labels, int max_iter, String model_path) {

    Mat labels_mat((int) features.size(), 1, CV_32SC1, labels.data());
    Mat data_mat = ConvertFeatures(features);

    printf("SVM Initialization\n");

    const double epsilon = exp(-6);

    Ptr<SVM> svm = initSVM(model_path, max_iter);

    printf("Ready...\n");
    svm->trainAuto(data_mat, ROW_SAMPLE, labels_mat);

    /*while(!svm->isTrained()) {
        printf("Training...\n");
        std::this_thread::sleep_for(2s);
    }*/

    printf("Finished.\n");

    svm->save(model_path);
}