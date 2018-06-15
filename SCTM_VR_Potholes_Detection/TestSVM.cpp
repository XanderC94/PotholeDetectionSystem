////
//// Created by Xander_C on 6/06/2018.
////
//
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include <opencv2/highgui.hpp>
//#include <opencv2/ml.hpp>
//
//using namespace cv;
//using namespace cv::ml;
//
//int main(int, char**)
//{
//    // Data for visual representation
//    int width = 512, height = 512;
//    Mat image = Mat::zeros(height, width, CV_8UC3);
//
//    // Set up training data
//    int labels[4] = { 1, -1, -1, -1 };
//    Mat labelsMat(4, 1, CV_32SC1, labels);
//
//    float trainingData[4][3] = { { 501.7234, 10.12454, 2.1343}, { 255.4455, 10.21343, 3.13431}, { 501.2343, 255.3333 , 4.34004}, { 10.0231, 501.0006, 0.43234} };
//    Mat trainingDataMat(4, 3, CV_32FC1, trainingData);
//
//    // Set up SVM's parameters
//    Ptr<SVM> svm = SVM::create();
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::RBF);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//
//    std::cout << trainingDataMat << std::endl << std::endl;
//
//    // Train the SVM with given parameters
//    Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
//    //svm->train(td);
//
//    // Or train the SVM with optimal parameters
//    svm->trainAuto(td);
//
//    svm->save("../test_model.yml");
//
////    Vec3b green(0, 255, 0), blue(255, 0, 0);
////    // Show the decision regions given by the SVM
////    for (int i = 0; i < image.rows; ++i)
////        for (int j = 0; j < image.cols; ++j)
////        {
////            Mat sampleMat = Mat_<float>(1, 2) << (j, i);
////            float response = svm->predict(sampleMat);
////
////            if (response == 1)
////                image.at<Vec3b>(i, j) = green;
////            else if (response == -1)
////                image.at<Vec3b>(i, j) = blue;
////        }
////
////    // Show the training data
////    int thickness = -1;
////    int lineType = 8;
////    circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
////    circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
////    circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
////    circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
////
////    // Show support vectors
////    thickness = 2;
////    lineType = 8;
////    Mat sv = svm->getSupportVectors();
////
////    for (int i = 0; i < sv.rows; ++i)
////    {
////        const float* v = sv.ptr<float>(i);
////        circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
////    }
////
////    imwrite("result.png", image);        // save the image
////
////    imshow("SVM Simple Example", image); // show it to the user
////    waitKey(0);
//
//}