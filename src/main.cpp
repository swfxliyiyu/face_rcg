//
// Created by yiyuli@pku.edu.cn on 17-5-9.
//

#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include "opencv2/objdetect.hpp"
#include "face_features/pre_treat.h"
#include "face_features/detect_face.h"
#include "face_features/load_data.h"
#include "face_features/test.h"

using namespace std;
using namespace cv;


string cascadeName = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "/opt/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string imgName = "../input/image_0001.png";





int main() {

    // 预处理图片
//    string dir_path = "../input/images/";
//    prt_images();
//    rectMeanStdDev(dir_path);
//    getMeanShape(dir_path);


    trainModel("../input/", 5, 5);


//    vector<string> img_names = getImgNames("../input/images/");
//    // 获取平均脸
//    vector<Point2f> mean_shape = load_features("../input/mean_shape.pts");
//    // 获取均值标准差map
//    map<string, float> mean_sdev;
//    ifstream in_stream("../input/mean_sdev.txt");
//    string s;
//    while (in_stream >> s) {
//        in_stream >> mean_sdev[s];
//    }
//    in_stream.close();
//    for (int k = 0; k < img_names.size(); ++k) {
//
//        string img_name = img_names[k];
//        cout << img_name << endl;
////        Mat temp = imread(string("../input/raw_images/") + "image_0078" + ".png");
////        vector<Point2f> ps = load_features(string("../input/raw_images/") + "image_0078" + ".pts");
//        Mat temp = imread(string("../input/images/") + img_name + ".jpg");
//        vector<Point2f> ps = load_features(string("../input/images/") + img_name + ".pts");
//
//        Rect featureRect = getFeatureRect(ps);
//        if (featureRect.x<0||featureRect.x+featureRect.width>=temp.cols
//                ||featureRect.y<0||featureRect.y+featureRect.height>=temp.rows) {
//            cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << img_name << endl;
//        }
//        vector<Rect> face_rects = detectFaceRect(temp, 2.0);
//        if (face_rects.empty()) cout << "no face rect" << endl;
//        Rect face_rect;
//        double diff = INT_MAX;
//        for (int j = 0; j < face_rects.size(); ++j) {
//            // 计算差距
//            Rect temp_rect = face_rects[j];
//            double temp_diff = pow((temp_rect.x - featureRect.x), 2)
//                               + pow((temp_rect.y - featureRect.y), 2);
//            // 若距离近则更新人脸框
//            face_rect = temp_diff < diff ? temp_rect : face_rect;
//            diff = temp_diff < diff ? temp_diff : diff;
//        }
//        for (int i = 0; i < face_rects.size(); ++i) {
//            rectangle(temp, face_rects[i], Scalar(255,0,255));
//        }
//
//        for (int j = 0; j < ps.size(); ++j) {
//            circle(temp, ps[j], 3, Scalar(255,255,0), -1, 8, 0);
//        }
//        imshow("", temp);
//        waitKey(0);
//    }


//    Mat m = imread("../input/testimages/image_0009.jpg");
//
//    test(m, "../input/", 1);
}
