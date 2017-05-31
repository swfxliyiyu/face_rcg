//
// Created by yiyuli@pku.edu.cn on 17-5-31.
//
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <opencv/cv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "map"
#include "test.cpp.h"
#include "load_data.h"
#include "detect_face.h"

using namespace std;
using namespace cv;

void test(Mat &img, string input_path, int itr) {
    // 获取平均脸
    vector<Point2f> mean_shape = load_features(input_path + "mean_shape.pts");
    // 获取均值标准差map
    map<string, float> mean_sdev;
    ifstream in_stream(input_path + "mean_sdev.txt");
    string s;
    while (in_stream >> s) {
        in_stream >> mean_sdev[s];
    }
    in_stream.close();
    vector<Rect> face_rects = detectFaceRect(img, 2.0);
    // 若检测不到人脸则返回
    if (face_rects.empty()) {
        cout << "图片" << "检测不到人脸!" << endl;
        return;
    }
    Rect &face_rect = face_rects[0];
    vector<Point2f> v0 = randomGenerFeartures(face_rect, mean_sdev,
                                              Size(img.rows, img.cols), mean_shape);
//    Rect frect = getFeatureRect(load_features("../input/001.pts"));
//    Rect v0rect = getFeatureRect(v0);
//    resizeFeatures(v0, frect.x - v0rect.x, frect.y - v0rect.y,
//                   (float)frect.width/v0rect.width, (float)frect.height/v0rect.height);
    Mat vk = generMatOfFeature(v0);
    Mat fk;
    Mat Rk;
    Mat pca_k;
    Mat ima;

    Mat im;

    img.copyTo(ima);
    img.copyTo(im);
    Mat m_k;
    for (int j = 0; j < vk.rows / 2; ++j) {
        Point2f p(vk.at<float>(2*j), vk.at<float>(2*j+1));
        circle(ima, p, 2, Scalar(0, 0, 255));
    }
    rectangle(ima, face_rect, Scalar(255,255,0));
    imshow("test", ima);
    waitKey();



    for (int i = 0; i < itr; ++i) {
        stringstream ss1;
        ss1 << "../output/model/R" << i << ".mdl";
        loadMat(ss1.str(), Rk);
        stringstream ss2;
        ss2 << "../output/model/p" << i << ".pca";
        loadMat(ss2.str(), pca_k);
        stringstream ss3;
        ss3 << "../output/model/p" << i << ".m";
        loadMat(ss3.str(), m_k);
        fk = computeSIFT(img, vk);
        cout << "fk" << fk << endl;
        fk = fk - m_k.t();
        gemm( pca_k, fk, 1, Mat(), 0, fk, 0 );
//        normalize(fk, fk);
        fk.push_back(1.f);


        cout << "Rk" << Rk << endl;
        cout << "fk" << fk << endl;
        Mat dk = Rk * fk;
        cout << "dk" << dk << endl;
        vk += dk;
        cout << "vk" << vk << endl;

        for (int j = 0; j < vk.rows / 2; ++j) {
            Point2f p(vk.at<float>(2*j), vk.at<float>(2*j+1));
            circle(im, p, 3, Scalar(0, 255, 0));
        }
        rectangle(im, face_rect, Scalar(255,255,0));
        imshow("test", im);
        waitKey();
    }


}