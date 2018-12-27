//
// Created by Xander_C on 27/06/2018.
//

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "../include/phdetection/ml_utils.hpp"

using namespace cv;
using namespace std;
using namespace phd::ontologies;

namespace phd {
    namespace ml {
        namespace utils {
            Mat ConvertFeatures(const vector<Features> &features) {

                Mat data(static_cast<int>( features.size()), 5, CV_32FC1);

                for (int i = 0; i < features.size(); i++) {
                    data.at<float>(i, 0) = features[i].averageGreyValue;
                    data.at<float>(i, 1) = features[i].contrast;
                    data.at<float>(i, 2) = features[i].skewness;
                    data.at<float>(i, 3) = features[i].energy;
                    data.at<float>(i, 4) = features[i].entropy;
                    //            data.at<float>(i, 5) = static_cast<float>(features[i].averageRGBValues.val[0]);
                    //            data.at<float>(i, 6) = static_cast<float>(features[i].averageRGBValues.val[1]);
                    //            data.at<float>(i, 7) = static_cast<float>(features[i].averageRGBValues.val[2]);
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

                Mat hogFeatures = ConvertHOGFeatures(features, (var_count - 1));

                Mat data(static_cast<int>( features.size()), 1 + hogFeatures.cols, CV_32FC1);

                for (int i = 0; i < features.size(); i++) {
                    data.at<float>(i, 0) = features[i].averageGreyValue;
                    for (int j = 1; j < hogFeatures.cols; ++j) {
                        data.at<float>(i, j) = hogFeatures.at<float>(i, j);
                    }
                }
                return data;
            }

            bool svmFindSomething(const Mat svmResult, int i) {
                return (svmResult.at<float>(0, i) == ClassificationClasses::pothole ||
                        svmResult.at<float>(0, i) == ClassificationClasses::asphaltCrack);
            }

            bool bayesFindSomething(const Mat bayesResult, int i) {
                return (bayesResult.at<int>(0, i) == ClassificationClasses::pothole ||
                        bayesResult.at<int>(0, i) == ClassificationClasses::asphaltCrack);
            }


            Mat mergeMultiClassifierResults(const Mat svmResult, const Mat bayesResult) {
                if (svmResult.size() != bayesResult.size()) {
                    cout << "Exception! The svmResult.size() must be equal to the bayesResult.size()" << endl;
                    exit(-1);
                }

                Mat result(svmResult.size(), CV_32SC1);

                for (int i = 0; i < svmResult.cols; i++) {
                    if (svmFindSomething(svmResult, i) && bayesFindSomething(bayesResult, i)) {
                        result.at<int>(0, i) = ClassificationClasses::pothole;
                    } else {
                        result.at<int>(0, i) = ClassificationClasses::streetSideWalkOrCar;
                    }
                }

                return result;
            }
        }

    }
}
