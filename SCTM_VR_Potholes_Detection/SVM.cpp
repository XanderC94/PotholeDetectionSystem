//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"

const double epsilon = exp(-6);

Mat Classifier(vector<Features> &features, int max_iter, String model_path){

    Mat labels_mat((int) features.size(), 1, CV_32SC1);

    Ptr<SVM> svm = SVM::create();

    svm->load(model_path);

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));

    double data[features.size()][5];

    for (int i = 0; i < features.size(); ++i) {

        Features f = features[i];
        data[i][0] = f.averageGrayValue.val[0];
        data[i][1] = f.contrast;
        data[i][2] = f.energy;
        data[i][3] = f.entropy;
        data[i][4] = f.skewness;
    }

    Mat data_mat((int) features.size(), 5, CV_32FC1 , data);

    svm->predict(data_mat, labels_mat);

    return labels_mat;
}

void Training(vector<Features> &features, vector<int> &labels, int max_iter, String model_path) {

    Ptr<SVM> svm = SVM::create();

    svm->load(model_path);

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, epsilon));

    double data[features.size()][5];

    for (int i = 0; i < features.size(); ++i) {

        Features f = features[i];
        data[i][0] = f.averageGrayValue.val[0];
        data[i][1] = f.contrast;
        data[i][2] = f.energy;
        data[i][3] = f.entropy;
        data[i][4] = f.skewness;
    }

    Mat labels_mat = Mat((int) features.size(), 1, CV_32SC1, labels.data());
    Mat data_mat((int) features.size(), 5, CV_32FC1 , data);

    Ptr<TrainData> td = TrainData::create(data_mat, ROW_SAMPLE, labels_mat);

    svm->trainAuto(td);

    while(!svm->isTrained()) {
        printf("Training...\n");
    }

    svm->save(model_path);
}