#include "CascadeClassifier.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

using namespace std;

int MyCascadeClassifier(const String img_path, const String cls) {

    const double V_offset = 1.00;
    const double H_offset = 0.00;
    const double Cutline_offset = 0.60;
    const double VP_offset_X = 0.50;
    const double VP_offset_Y = 0.55;

    vector<Rect> output;
    Size scale(640, 480);

    Mat img0 = imread(img_path, IMREAD_COLOR), img1;

    cvtColor(img0, img0, CV_RGB2GRAY);

    img0.convertTo(img1, CV_8U);

    resize(img1, img0, scale);

    img0 = img0(Rect(Point(0, img0.rows*Cutline_offset), Point(img0.cols - 1, img0.rows - 1)));

    CascadeClassifier cascade;
    bool b = cascade.load(cls);

    clog << "Loaded casced classifier @" << cls << " : " << (b == 0 ? "False" : "True") << endl;

    cascade.detectMultiScale(img0, output);

    clog << "Detected " << output.size() << " objects as potholes..." << endl;

    for (auto &r : output) {
    rectangle(img0, r, Scalar(255, 0, 0));
    }

    imshow("Pothole Detection", img0);

    waitKey();

    return 1;
}//
// Created by Xander_C on 24/03/2018.
//

