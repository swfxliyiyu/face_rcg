//
// Created by swfxliyiyu on 17-5-9.
//

#include "pre_treat.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include "opencv2/contrib/contrib.hpp"
#include "load_data.h"
#include <boost/regex.hpp>

Rect getFeatureRect(const vector<Point2f> &vector);

void saveFeatures(vector<Point2f> &vector, string path, string image_name);


using namespace std;
using namespace cv;



void resizeFeatures(vector<Point2f> &features, int x_move, int y_move, float w_scale, float h_scale) {

    for (vector<Point2f>::iterator itr = features.begin(); itr < features.end(); ++itr) {
        // 旧特征点坐标
        float x = itr->x;
        float y = itr->y;
        // 先移动再缩放
        float n_x = (x + x_move) * w_scale;
        float n_y = (y + y_move) * h_scale;
        // 赋值
        itr->x = n_x;
        itr->y = n_y;
    }

}

void translate(cv::Mat const& src, cv::Mat& dst, int dx, int dy)
{
    CV_Assert(src.depth() == CV_8U);
    const int rows = src.rows;
    const int cols = src.cols;
    dst.create(rows, cols, src.type());
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //平移后坐标映射到原图像
            int x = j - dx;
            int y = i - dy;

            //保证映射后的坐标在原图像范围内
            if (x >= 0 && y >= 0 && x < cols && y < rows)
                for (int k = 0; k < 3; ++k) {
                    dst.at<Vec3b>(i, j)[k] = src.at<Vec3b>(y, x)[k];
                }
        }
    }
}

void resizeAndSave(Mat &src_img, vector<Point2f> &features, string img_name, int t_w, int t_h) {

    // 输出地址
    string output_path = "/home/swfxliyiyu/CLionProjects/face_rcg/input/images/";
//    // 判断文件是否存在
//    fstream file;
//    file.open(output_path + img_name + ".jpg", ios::in);
//    if (access((output_path + img_name + ".jpg").c_str(), 0) != -1) {
//        cout << img_name << "已存在" << endl;
//        return;
//    }

    // 特征点包围盒
    Rect rect = getFeatureRect(features);
    // 计算新图片位置
    int x = rect.x - rect.width / 4;
    int y = rect.y - rect.height / 2;
    int w = rect.width * 3/2;
    int h = rect.height * 2;
//    // 如果越界则不作为训练样本
//    if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
//        std::cout << img_name << "采样越界，不作为训练样本" << endl;
//        cout << "x:" << x << endl;
//        cout << "y:" << y << endl;
//        return;
//    }
    // 如果越界则平移图片
    if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
        std::cout << img_name << "采样越界，进行平移" << endl;
        cout << "x:" << x << endl;
        cout << "y:" << y << endl;
        cout << "x+w:" << x+w << endl;
        cout << "y+h:" << y+h << endl;
        cout << "cols:" << src_img.cols << endl;
        cout << "rows:" << src_img.rows << endl;
        int x_move;
        int y_move;
        if (x < 0) {
            x_move = -x + 1;
            x += x_move;
        } else if ( x + w >= src_img.cols ) {
            x_move = src_img.cols - x - w - 1;
            x += x_move;
        }
        if (y < 0) {
            y_move = -y + 1;
            y += y_move;
        } else if ( y + h >= src_img.rows ){
            y_move = src_img.rows - x - w - 1;
            y += y_move;
        }
        // 变换图片
        Mat temp;
        translate(src_img, temp, x_move, y_move);
        src_img = temp;
        // 变换特征
        resizeFeatures(features, x_move, y_move, 1, 1);
        // 若变换后依然越界，则不做处理
        if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
            std::cout << img_name << "平移后仍越界，不作为训练样本" << endl;
            cout << "x:" << x << endl;
            cout << "y:" << y << endl;
            cout << "x+w:" << x+w << endl;
            cout << "y+h:" << y+h << endl;
            cout << "cols:" << src_img.cols << endl;
            cout << "rows:" << src_img.rows << endl;
            return;
        }
    }
    // 获取新图片
    Mat small_img(src_img, Range(y, y + h), Range(x, x + w));

    // 调整大小
    resize(small_img, small_img, Size(t_w, t_h));
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
    imwrite(dst_path, small_img);

}


Rect getFeatureRect(const vector <Point2f> &features) {
    // 初始化特征框的上下左右边界
    int top = INT_MAX, bot = 0,
            left = INT_MAX, right = 0;
    // 获取边界
    for (vector<Point2f>::const_iterator i = features.begin(); i != features.end(); ++i) {
        float x = i->x, y = i->y;
        if (y < top) top = (int) floor(y);
        if (y > bot) bot = (int) ceil(y);
        if (x < left) left = (int) floor(x);
        if (x > right) right = (int) ceil(x);
    }
    return Rect(left, top, right - left, bot - top);
}

void saveFeatures(std::vector<Point2f> &features, string path, string image_name) {

    ofstream output(path);
    output << "image_name: " << image_name << endl;
    output << "n_points: " << features.size() << endl;
    output << "{" << endl;
    for (int i = 0; i < features.size(); ++i) {
        Point2f point = features[i];
        output << point.x << " " << point.y << endl;
    }
    output << "}" << endl;
    output.close();

}

vector<string> getFiles(string dirname){
    DIR *dp;
    struct dirent *dirp;

    // 存储文件名
    vector<string> file_names;

    if((dp=opendir(dirname.c_str()))==NULL){
        perror("opendir error");
        exit(1);
    }

    while((dirp=readdir(dp))!=NULL){
        if((strcmp(dirp->d_name,".")==0)||(strcmp(dirp->d_name,"..")==0))
            continue;
        file_names.push_back(string(dirp->d_name));
        //printf("%6d:%-19s %5s\n",dirp->d_ino,dirp->d_name,(dirp->d_type==DT_DIR)?("(DIR)"):(""));
    }

    return file_names;
}

void prt_images() {
    string dir_path = "../input/raw_images/";
    vector<string> files = getFiles(dir_path);
    // 删除目录中后缀为pts的
    for (vector<string>::iterator i = files.begin(); i < files.end(); ++i) {
        string::iterator sit = --(i->end());
        if (*sit != 'g')
            files.erase(i);
    }
    // 对于每个.png文件，提取文件名，处理图片
    for (vector<string>::iterator i = --files.end(); i >= files.begin(); --i) {
        int cnt = 0;
        // 读取图片
        Mat img = imread(dir_path + (*i));
        for (string::iterator it = --(i->end()); cnt < 4; ++cnt, --it) {
            i->erase(it);
        }
        // 读取特征
        vector<Point2f> f = load_features(dir_path + (*i) + ".pts");
        // 处理图片
        resizeAndSave(img, f, *i, 400, 450);
        
    }

}

