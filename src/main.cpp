//
// Created by yiyuli@pku.edu.cn on 17-5-9.
//

#include <iostream>
#include <fstream>
#include "opencv2/objdetect.hpp"
#include "face_features/pre_treat.h"
#include "face_features/detect_face.h"
#include "face_features/load_data.h"

using namespace std;
using namespace cv;


string cascadeName = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "/opt/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string imgName = "../input/image_0001.png";



void something(string input_path, int itr, int n_vk) {
    class ImgInfo {
    public:

        // vk向量集合
        vector<Mat> vector_vk;
        // fk向量集合（增广）
        vector<Mat> vector_fk;
        // dk向量集合
        vector<Mat> vector_dk;
        // 特征点向量
        Mat fea;

    };

    // 存储图片和图片矩阵
    map<string, ImgInfo> img_map;

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
    // 获取图片名
    vector<string> img_names = getImgNames(input_path + "images/");
    // 处理每个图片名，生成特征向量x0
    Mat pca_fk_mat(0, 0, CV_32F);
    for (int i = 0; i < img_names.size(); ++i) {
        // 获取图片名
        string img_name = img_names[i];
        cout << "正在生成图片:" << img_name << "的初始特征点..." << endl;
        // 将图片及其矩阵加入map
        img_map[img_name] = ImgInfo();
        ImgInfo &imgInfo = img_map[img_name];
        // 读取图片
        Mat img = imread(input_path + "images/" + img_name + ".jpg");
        // 读取特征点
        vector<Point2f> fea = load_features(input_path + "images/" + img_name + ".pts");
        Rect featureRect = getFeatureRect(fea);
        Mat fea_mat = generMatOfFeature(fea);
        imgInfo.fea = fea_mat;
        // 检测人脸框
        cout << "检测人脸框..." << endl;
        vector<Rect> face_rects = detectFaceRect(img, 2.0);
        // 取与特征点最近的一张人脸框
        // TODO
        if (face_rects.empty()) {
            cout << "图片" << img_name << "检测不到人脸!" << endl;
            img_map.erase(img_name);
            continue;
        }
        Rect face_rect = face_rects[0];
        double diff = INT_MAX;
        for (int j = 0; j < face_rects.size(); ++j) {
            // 计算差距
            Rect temp_rect = face_rects[j];
            double temp_diff = pow((temp_rect.x - featureRect.x), 2)
                               + pow((temp_rect.y - featureRect.y), 2);
            // 若距离近则更新人脸框
            face_rect = temp_diff < diff ? temp_rect : face_rect;
            diff = temp_diff < diff ? temp_diff : diff;
        }
        // 对于每张图片生成n_vk个v0向量

//        rectangle(img, face_rect, Scalar::all(0));
        for (int k = 0; k < n_vk; ++k) {
            cout << "生成第" << k + 1 << "个v0向量..." << endl;

            vector<Point2f> v0 = randomGenerFeartures(face_rect, mean_sdev,
                                                      Size(img.rows, img.cols), mean_shape);
            // TODO
//            for (int j = 0; j < v0.size(); ++j) {
//                circle(img, v0[j], 2, Scalar(255, 0, 255), -1, 8, 0);
//            }
//            imshow("", img);
//            waitKey();
            Mat v0_mat = generMatOfFeature(v0);
            Mat d0_mat = fea_mat - v0_mat;
            imgInfo.vector_vk.push_back(v0_mat);
            imgInfo.vector_dk.push_back(d0_mat);
            // f0是增广的
            cout << "计算SIFT..." << endl;
            Mat f0_mat = computeSIFT(img, v0);
            cout << f0_mat.t() << endl;
//            f0_mat.push_back(1.f);
//            imgInfo.vector_fk.push_back(f0_mat);
            pca_fk_mat.push_back(f0_mat.t());
        }
    }
    // 计算fk均值并保存
    Mat mean(0, 0, CV_32F);
    for (int l = 0; l < pca_fk_mat.cols; ++l) {
        Scalar scalar;
        meanStdDev(pca_fk_mat.col(l), scalar, Scalar());
        mean.push_back(scalar[0]);
    }
    ofstream ofstream1("../output/model/p0.m");
    mean = mean.t();
    ofstream1 << mean.rows << " " << mean.cols << endl;
    ofstream1 << mean;
}

int main() {

//    prt_images();
    string dir_path = "../input/images/";
//    vector<string> files = getFiles(dir_path);
//    for (vector<string>::iterator i = --files.end(); i >= files.begin(); --i) {
//        string::iterator sit = --(i->end());
//        if (*sit != 'g') {
//            files.erase(i);
//        }
//    }
//
//    for (vector<string>::iterator i = files.begin(); i < files.end(); ++i) {
//        cout << *i << endl;
//    }
//    // 对于每个.png文件，提取文件名，处理图片
//    for (vector<string>::iterator i = files.begin(); i < files.end(); ++i) {
//        Mat prt_img = imread(dir_path + *i);
//        cout << dir_path + *i << endl;
//        int cnt = 0;
//        for (string::iterator it = --(i->end()); cnt < 4; ++cnt, --it) {
//            i->erase(it);
//        }
//        vector<Point2f> prt_features = load_features(dir_path + *i + ".pts");
//        for (int j = 0; j < prt_features.size(); ++j) {
//            Point2f feature = prt_features.at(j);
//            circle(prt_img, feature, 5, Scalar(0, 255, 0), -1, 8, 0);
//        }
//        imshow("", prt_img);
//        waitKey(0);
//    }
//    rectMeanStdDev(dir_path);
//    getMeanShape(dir_path);
//    Mat img = imread("../input/raw_images/image_0001.png");
//    vector<Point2f> feas = load_features("../input/raw_images/image_0001.pts");
    trainModel("../input/", 5, 5);

//    something("../input/", 5, 5);
//    Mat m = imread("../input/images/image_0044.jpg");
//
//    test(m, "../input/", 5);


}
