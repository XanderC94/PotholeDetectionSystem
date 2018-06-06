//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"

Ptr<SVM> initSVM(String model_path, int max_iter, bool isTraining = false) {
    const double epsilon = exp(-6);

    Ptr<SVM> svm = SVM::create();

    if (!isTraining) {
        try {
            svm = svm->load(model_path);
        } catch (Exception ex) {
            printf(ex.msg.c_str());
        }
    }

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));
}

Mat ConvertFeatures(vector<Features> &features) {

    double data[features.size()][5];

    for (int i = 0; i < features.size(); ++i) {

        Features f = features[i];
        data[i][0] = f.averageGrayValue;
        data[i][1] = f.contrast;
        data[i][2] = f.energy;
        data[i][3] = f.entropy;
        data[i][4] = f.skewness;
    }

    return Mat((int) features.size(), 5, CV_32FC1 , data);
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

    Ptr<SVM> svm = initSVM(model_path, max_iter, true);

    Ptr<TrainData> td = TrainData::create(data_mat, ROW_SAMPLE, labels_mat);

    svm->trainAuto(td);

    while(!svm->isTrained()) {
        printf("Training...\n");
    }

    svm->save(model_path);
}