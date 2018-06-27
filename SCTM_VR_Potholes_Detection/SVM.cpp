//
// Created by Xander_C on 5/06/2018.
//

#include "SVM.h"
#include <opencv2/ml.hpp>

using namespace cv::ml;

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

Mat ConvertFeatures(const vector<Features> &features) {

    Mat data((int) features.size(), 5, CV_32FC1);

    for (int i = 0; i < features.size(); i++) {
        data.at<float>(i, 0) = features[i].averageGreyValue;
        data.at<float>(i, 1) = features[i].contrast;
        data.at<float>(i, 2) = features[i].skewness;
        data.at<float>(i, 3) = features[i].energy;
        data.at<float>(i, 4) = features[i].entropy;

//        cout << data.row(i) << endl;
    }

    return data;
}

Mat ConvertHOGFeatures(const vector<Features> &features) {

    // Find max
    int max_length = 0;
    for (auto ft : features) {
        if (ft.hogDescriptors.cols > max_length) {
            max_length = ft.hogDescriptors.cols;
        }
    }

    cout << max_length << endl;

    Mat hog = Mat::zeros(features.size(), max_length, CV_32FC1);

    for (int i = 0; i < features.size(); ++i) {
        const int L = features[i].hogDescriptors.cols;
        for (int j = 0; j < L; ++j) {
            hog.at<float>(i, j) = features[i].hogDescriptors.at<float>(0, j);
        }
    }

    return hog;
}

void Classifier(const vector<Features> &features, const int max_iter, const string model_path, Mat &labels){

    Mat data_mat = ConvertHOGFeatures(features);

    Ptr<SVM> svm = initSVM(model_path, max_iter);

    auto flag = svm->predict(data_mat, labels);

//    cout << "FLAG: " << flag << endl;
    transpose(labels, labels);
    cout << labels << endl;
}

void Training(const vector<Features> &features, const Mat &labels, const int max_iter, const string model_path) {

//    Mat data_mat = ConvertFeatures(features);

    Mat data_mat = ConvertHOGFeatures(features);

    cout << "HOG FT size " << data_mat.rows << "*" << data_mat.cols << endl;

    printf("SVM Initialization\n");

    Ptr<SVM> svm = initSVM(model_path, max_iter);

    printf("Ready...\n");

    auto data = TrainData::create(data_mat, ROW_SAMPLE, labels);

    svm->trainAuto(data, 3);

    printf("Finished.\n");

    svm->save(model_path);
}