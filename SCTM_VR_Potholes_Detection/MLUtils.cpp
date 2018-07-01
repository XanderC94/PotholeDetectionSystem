//
// Created by Xander_C on 27/06/2018.
//

#include "SVM.h"
#include "MLUtils.h"

namespace mlutils {
    Mat ConvertFeatures(const vector<Features> &features) {

        Mat data((int) features.size(), 8, CV_32FC1);

        for (int i = 0; i < features.size(); i++) {
            data.at<float>(i, 0) = features[i].averageGreyValue;
            data.at<float>(i, 1) = features[i].contrast;
            data.at<float>(i, 2) = features[i].skewness;
            data.at<float>(i, 3) = features[i].energy;
            data.at<float>(i, 4) = features[i].entropy;
            data.at<float>(i, 5) = static_cast<float>(features[i].averageRGBValues.val[0]);
            data.at<float>(i, 6) = static_cast<float>(features[i].averageRGBValues.val[1]);
            data.at<float>(i, 7) = static_cast<float>(features[i].averageRGBValues.val[2]);
//        cout << data.row(i) << endl;
        }

        return data;
    }

    Mat ConvertHOGFeatures(const vector<Features> &features, const int var_count) {

        // Find max
        int max_length = var_count > 0 ? var_count : 0;

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


    Mat ConvertFeaturesForBayes(const vector<Features> &features) {
        return ConvertFeatures(features);
    }

    Mat ConvertFeaturesForSVM(const vector<Features> &features, const int var_count) {

        Mat hogFeatures = ConvertHOGFeatures(features,(var_count - 1));

        Mat data((int) features.size(), 1 + hogFeatures.cols, CV_32FC1);

        for (int i = 0; i < features.size(); i++) {
            data.at<float>(i, 0) = features[i].averageGreyValue;
            for (int j = 1; j < hogFeatures.cols; ++j) {
                data.at<float>(i, j) = hogFeatures.at<float>(i, j);
            }
        }
        return data;
    }

}
