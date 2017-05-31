//
// Created by yiyuli@pku.edu.cn on 17-5-31.
//
#include "test.h"
#include <opencv2/imgcodecs.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <opencv2/features2d.hpp>
#include "detect_face.h"
#include "pre_treat.h"
#include "load_data.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv/cv.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <set>
#include <random>

using namespace std;
using namespace cv;



void resizeTestImg(Mat &src_img, vector<Point2f> &features, Rect face_rect, string img_name, int t_w, int t_h) {

    // 输出地址
    string output_path = "../input/";
//    // 判断文件是否存在
//    fstream file;
//    file.open(output_path + img_name + ".jpg", ios::in);
//    if (access((output_path + img_name + ".jpg").c_str(), 0) != -1) {
//        cout << img_name << "已存在" << endl;
//        return;
//    }


//    // 特征点包围盒
//    Rect rect = getFeatureRect(features);
    // 计算新图片位置
    int x = (float) face_rect.x - face_rect.width / 2;
    int y = (float) face_rect.y - face_rect.height / 2;
    int w = (float) face_rect.width * 2;
    int h = (float) face_rect.height * 2;
    // 获取新图片
    Mat small_img(src_img, Range(y<0?0:y, y+h>=src_img.rows?src_img.rows-1:y+h-1),
                  Range(x<0?0:x, x+w>=src_img.cols?src_img.cols-1:x+w-1));
    Mat rect_img = Mat::zeros(h, w, CV_8UC3);
    small_img.copyTo(rect_img.rowRange(y<0?-y:0,
                                       y+h>=src_img.rows?src_img.rows-y-1:h-1).colRange(
            x<0?-x:0, x+w>=src_img.cols?src_img.cols-x-1:w-1));

    // 调整大小
    resize(rect_img, rect_img, Size(t_w, t_h));
    // 调整特征点大小
    int x_move = -x;
    int y_move = -y;
    float w_scale = (float) t_w / w;
    float h_scale = (float) t_h / h;
    resizeFeatures(features, x_move, y_move, w_scale, h_scale);
    string feature_path = output_path + img_name + ".pts";
    saveFeatures(features, feature_path, img_name);


    // 保存图片
    string dst_path = output_path + img_name + ".jpg";
    imwrite(dst_path, rect_img);
}