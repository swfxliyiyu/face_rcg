//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

#include <opencv2/imgcodecs.hpp>
#include "detect_face.h"
#include "pre_treat.h"
#include "load_data.h"

using namespace std;
using namespace cv;

/**
 * 检测人脸方框
 * @param img 输入图片
 * @param scale 图片放大规模
 * @return 检测到的人脸框
 */
vector<Rect> detectFaceRect(Mat &img, double scale = 1.0) {
    // 分类器
    CascadeClassifier cascade;
    cascade.load("/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
    // 检测时间
    double t = 0;
    // 存储检测到的脸部方框
    vector<Rect> faces, result_faces;

    Mat gray, smallImg;

    // 将原图转换为灰度图
    cvtColor( img, gray, COLOR_BGR2GRAY );
    // 缩放规模
    double fx = 1 / scale;
    // 缩放图像
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    // 增强图片对比度
    equalizeHist( smallImg, smallImg );
    // 计算时钟
    t = (double)getTickCount();
    // 从缩放后的灰度图中检测人脸
    cascade.detectMultiScale( smallImg, faces,
                              1.1, 2, 0
                                      //|CASCADE_FIND_BIGGEST_OBJECT
                                      //|CASCADE_DO_ROUGH_SEARCH
                                      |CASCADE_SCALE_IMAGE,
                              Size(30, 30));

    t = (double)getTickCount() - t;
    // 输出检测时间
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    // 迭代检测到的脸
    for ( size_t i = 0; i < faces.size(); i++ ) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;

        Rect rect;

        int radius;

        double aspect_ratio = (double)r.width/r.height;
        // 转换为原图中的坐标
        {
            int n_x = cvRound(r.x * scale);
            int n_y = cvRound(r.y * scale);
            int n_w = r.width * scale;
            int n_h = r.width * scale;
            rect = Rect(n_x, n_y, n_w, n_h);
            result_faces.push_back(rect);
        }

    }
    return result_faces;
}

/**
 * 获取特征框和人脸检测框的均差和均差方差
 * return int[x_mean, y_mean, w_mean, h_mean, x_var, y_var, w_var, h_var]
 */
int* rectMeanVar(string pts_path) {

    int mean_var[8];
    vector<string> file_names = getImgNames(pts_path);
    int file_count = file_names.size();
    for (int i = 0; i < file_count; ++i) {
        string file_name = file_names[i];
        vector<Point2f> features = load_features(pts_path + file_name + ".pts");
        Rect featureRect = getFeatureRect(features);
        Mat img = imread(pts_path + file_name + "jpg");
        vector<Rect> face_rects = detectFaceRect(img);
        // 取与特征点最近的一张人脸框
        Rect face_rect;
        for (int j = 0; j < face_rects.size(); ++j) {
            //TODO
        }
    }
    return mean_var;
}
