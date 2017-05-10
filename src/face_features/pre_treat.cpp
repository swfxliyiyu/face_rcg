//
// Created by swfxliyiyu on 17-5-9.
//

#include "pre_treat.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv/cv.hpp>
#include "detect_face.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

Rect getFeatureRect(const vector<Point2f> &vector);

using namespace std;
using namespace cv;


void resizeAndSave(Mat& src_img, const vector<Point2f>& features, string img_name, int t_w, int t_h) {

    // 特征点包围盒
    Rect rect = getFeatureRect(features);
    // 计算新图片位置
    int x = rect.x - rect.width / 2;
    int y = rect.y - rect.height / 2;
    int w = rect.width * 2;
    int h = rect.height * 2;
    // 如果越界则不作为训练样本
    if (x<0 || y<0 || x+w >= src_img.cols || y+h >= src_img.rows) {
        std::cout << img_name << "采样越界，不作为训练样本" << endl;
        cout << "x:" << x << endl;
        cout << "y:" << y << endl;
        return;
    }
    // 获取新图片
    Mat small_img(src_img, Range(y, y+h), Range(x, x+w));
    // 调整大小
    resize(small_img, small_img, Size(t_w, t_h));
    //TODO 调整特征点大小

    // 保存图片
    string dst_path = "/home/swfxliyiyu/CLionProjects/face_rcg/output/images/" + img_name + ".jpg";
    imwrite(dst_path, small_img);

}

Rect getFeatureRect(const vector<Point2f> &features) {
    // 初始化特征框的上下左右边界
    int top = INT_MAX, bot = 0,
            left = INT_MAX, right = 0;
    // 获取边界
    for (vector<Point2f>::const_iterator i = features.begin(); i != features.end(); ++i) {
        float x = i->x, y = i->y;
        if (y < top) top = (int)floor(y);
        if (y > bot) bot = (int)ceil(y);
        if (x < left) left = (int)floor(x);
        if (x > right) right = (int)ceil(x);
    }
    return Rect(left, top, right - left, bot - top);
}

void resizeFeatures(vector<Point2f>& features, int x_move, int y_move, float w_scale, float h_scale) {

    for (vector<Point2f>::iterator itr = features.begin(); itr < features.end(); ++itr) {
        // 旧特征点坐标
        float x = itr->x;
        float y = itr->y;
        // 先移动再缩放
        float n_x = (x + x_move)*w_scale;
        float n_y = (y + y_move)*h_scale;
        // 赋值
        itr->x = n_x;
        itr->y = n_y;
    }

}

