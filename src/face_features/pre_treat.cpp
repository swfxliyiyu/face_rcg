//
// Created by yiyuli@pku.edu.cn on 17-5-9.
//

#include "pre_treat.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include "opencv2/contrib/contrib.hpp"
#include "load_data.h"
#include "detect_face.h"
#include <boost/regex.hpp>
#include <opencv/cv.hpp>


void saveFeatures(vector<Point2f> &vector, string path, string image_name);


Mat rot_scale_align(Mat mat, float cx, float cy, int n);

using namespace std;
using namespace cv;

/**
 * 调整特征点坐标
 * @param features 输入特征点数组
 * @param x_move 移动横坐标
 * @param y_move 移动纵坐标
 * @param w_scale 宽度变换规模
 * @param h_scale 高度变换规模
 */
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

void translate(cv::Mat const &src, cv::Mat &dst, int dx, int dy) {
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

/**
 * 调整图片和特征点大小
 * @param src_img 输入图片
 * @param features 特征点数组
 * @param img_name 图片名称
 * @param t_w 目标图片宽度
 * @param t_h 目标图片高度
 */
void resizeAndSave(Mat &src_img, vector<Point2f> &features, Rect face_rect, string img_name, int t_w, int t_h) {

    // 输出地址
    string output_path = "../input/images/";
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
//    // 如果越界则不作为训练样本
//    if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
//        std::cout << img_name << "采样越界，不作为训练样本" << endl;
//        cout << "x:" << x << endl;
//        cout << "y:" << y << endl;
//        return;
//    }
    // 如果越界则平移图片
//    if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
//        std::cout << img_name << "采样越界，进行平移" << endl;
//        cout << "x:" << x << endl;
//        cout << "y:" << y << endl;
//        cout << "x+w:" << x + w << endl;
//        cout << "y+h:" << y + h << endl;
//        cout << "cols:" << src_img.cols << endl;
//        cout << "rows:" << src_img.rows << endl;
//        int x_move = 0;
//        int y_move = 0;
//        if (x < 0) {
//            x_move = -x + 1;
//            x += x_move;
//        } else if (x + w >= src_img.cols) {
//            x_move = src_img.cols - x - w - 1;
//            x += x_move;
//        }
//        if (y < 0) {
//            y_move = -y + 1;
//            y += y_move;
//        } else if (y + h >= src_img.rows) {
//            y_move = src_img.rows - x - w - 1;
//            y += y_move;
//        }
//        // 变换图片
//        Mat temp;
//        translate(src_img, temp, x_move, y_move);
//        src_img = temp;
//        // 变换特征
//        resizeFeatures(features, x_move, y_move, 1, 1);
//        // 若变换后依然越界，则不做处理
//        if (x < 0 || y < 0 || x + w >= src_img.cols || y + h >= src_img.rows) {
//            std::cout << img_name << "平移后仍越界，不作为训练样本" << endl;
//            cout << "x:" << x << endl;
//            cout << "y:" << y << endl;
//            cout << "x+w:" << x + w << endl;
//            cout << "y+h:" << y + h << endl;
//            cout << "cols:" << src_img.cols << endl;
//            cout << "rows:" << src_img.rows << endl;
//            return;
//        }
//    }
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

/**
 * 获取指定特征点的包围框
 * @param features 特征点数组
 * @return
 */
Rect getFeatureRect(const vector<Point2f> &features) {
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

/**
 * 保存特征点
 * @param features 特征点数组
 * @param path 保存路径（包括后缀名）
 * @param image_name 图片名称
 */
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

/**
 * 获取图片名称（不包含后缀名）
 * @param dir_path 图片路径
 * @return vector 图片名称数组
 */
vector<string> getImgNames(string dir_path) {
    vector<string> files = getFiles(dir_path);
    // 提取目录中的不重复的图片名
    for (vector<string>::iterator i = --files.end(); i >= files.begin(); --i) {
        string::iterator sit = --(i->end());
        if (*sit != 'g')
            files.erase(i);
        else {
            // 删除后缀名
            int cnt = 0;
            for (string::iterator it = --(i->end()); cnt < 4; ++cnt, --it) {
                i->erase(it);
            }
        }
    }
    return files;
}

/**
 * 提取图片中包含人脸中的一部分，并对图片和相应特征点进行缩放
 */
void prt_images() {
    string dir_path = "../input/raw_images/";
    vector<string> files = getImgNames(dir_path);
    // 对于每个文件名，处理图片
    for (vector<string>::iterator i = --files.end(); i >= files.begin(); --i) {
        // 读取图片
        Mat img = imread(dir_path + (*i) + ".png");
        cout << "正在预处理图片:" << *i << "..." << endl;
        // 读取特征
        vector<Point2f> f = load_features(dir_path + (*i) + ".pts");
        Rect featureRect = getFeatureRect(f);
        // 人脸包围框
        vector<Rect> face_rects = detectFaceRect(img, 2.0);
        // 如果图片检测不到人脸则跳过
        if (face_rects.empty()) continue;
        // 取与特征点最近的一张人脸框
        Rect face_rect;
        double diff = INT_MAX;
        for (int j = 0; j < face_rects.size(); ++j) {
            // 计算差距
            Rect temp_rect = face_rects[j];
            double temp_diff = pow((temp_rect.x - featureRect.x), 2)
                               + pow((temp_rect.y - featureRect.y), 2)
            + pow(temp_rect.width - featureRect.width, 2)
            + pow(temp_rect.height - featureRect.height, 2);
            // 若距离近则更新人脸框
            face_rect = temp_diff < diff ? temp_rect : face_rect;
            diff = temp_diff < diff ? temp_diff : diff;
        }
        // 处理图片
        resizeAndSave(img, f, face_rect, *i, 400, 400);
    }
}


/**
 * 普氏分析
 * @param X
 * @param itol
 * @param ftol
 * @return
 */
Mat procrustes(const Mat &X, const int itol, const float ftol) {

    // X.cols:特征点个数N，X.rows: 图像数量*2
    int N = X.cols, n = X.rows / 2;
    //remove centre of mass
    Mat P = X.clone();
    for (int i = 0; i < N; i++) {
        /*取X第i个列向量*/
        Mat p = P.col(i);
        float mx = 0, my = 0;
        for (int j = 0; j < n; j++) {
            mx += p.at<float>(2 * j);
            my += p.at<float>(2 * j + 1);
        }

        /*分别求图像集，2维空间坐标x和y的平均值*/
        mx /= n;
        my /= n;
        /*对x,y坐标去中心化*/
        for (int j = 0; j < n; j++) {
            p.at<float>(2 * j, 0) -= mx;
            p.at<float>(2 * j + 1, 0) -= my;
        }
    }
    //optimise scale and rotation
    Mat C_old;
    for (int iter = 0; iter < itol; iter++) {
        // 计算（含n个形状）的重心
        Mat C = P * Mat::ones(N, 1, CV_32F) / N;
        //C为2n*1维矩阵，含n个重心，对n个重心归一化处理
        normalize(C, C);
        if (iter > 0) {
            if (norm(C, C_old) < ftol)//norm:求绝对范数，小于阈值，则退出循环
                break;
        }
        C_old = C.clone();
        for (int i = 0; i < N; i++) {
            //求当前形状与归一化重心之间的旋转角度
            Mat R = rot_scale_align(P.col(i), C.at<float>(i * 2, n), C.at<float>(i * 2 + 1), n);
            for (int j = 0; j < n; j++) {
                float x = P.at<float>(2 * j, i), y = P.at<float>(2 * j + 1, i);
                /*仿射变化*/
                P.at<float>(2 * j, i) = R.at<float>(0, 0) * x + R.at<float>(0, 1) * y;
                P.at<float>(2 * j + 1, i) = R.at<float>(1, 0) * x + R.at<float>(1, 1) * y;
            }
        }
    }
    return P;
}

/**
 * 普氏分析中用于提取旋转矩阵
 * @param mat
 * @param cx
 * @param cy
 * @param n
 * @return
 */
Mat rot_scale_align(Mat mat, float cx, float cy, int n) {
    // rot为旋转矩阵
    Mat rot;
    rot.create(2, 2, CV_32F);
    // 计算矩阵元素
    float a = 0, b = 0;
    float sum_square = 0;
    for (int i = 0; i < n; ++i) {
        // 第i个图像的特征点x，y。求x*cx + y*cy
        a += mat.at<float>(2 * i) * cx + mat.at<float>(2 * i + 1) * cy;
        // 第i个图像的特征点x，y。求x*cx - y*cy
        b += mat.at<float>(2 * i) * cx - mat.at<float>(2 * i + 1) * cy;
        // 求x,y平方和
        sum_square += pow(mat.at<float>(2 * i), 2) + pow(mat.at<float>(2 * i + 1), 2);
    }
    a /= sum_square;
    b /= sum_square;
    // 赋值矩阵
    rot.at<float>(0, 0) = a;
    rot.at<float>(0, 1) = -b;
    rot.at<float>(1, 0) = b;
    rot.at<float>(1, 1) = a;
    return rot;
}


/**
 * 获取平均脸
 * @param dirpath 特征点目录
 */
void getMeanShape(string dirpath) {

    // 获取图片名
    vector<string> files = getImgNames(dirpath);
    // faces用于存所有人脸特征，规模为2n*N，n为图像数量，N为特征点数量
    Mat faces;
    vector<Point2f> f = load_features(dirpath + files[0] + ".pts");
    faces.create((int) (files.size() * 2), (int) f.size(), CV_32F);
    // 载入特征点
    for (int i = 0; i < files.size(); ++i) {
        string filename = files[i];
        vector<Point2f> face = load_features(dirpath + filename + ".pts");
        Mat mat_x;
        Mat mat_y;

        mat_x.create(1, (int) (face.size()), CV_32F);
        mat_y.create(1, (int) (face.size()), CV_32F);
        for (int j = 0; j < face.size(); ++j) {
            mat_x.at<float>(j) = face[j].x;

            mat_y.at<float>(j) = face[j].y;
        }
        mat_x.copyTo(faces.row(2 * i));
        mat_y.copyTo(faces.row(2 * i + 1));

    }

//    普氏分析失败，直接求平均效果不错
//    // 对特征点进行普氏分析
//    faces = procrustes(faces, 10, 0.1);
//
//    faces *= 1e16;
    // 求平均
    vector<Point2f> mean_shape;
    mean_shape.resize(f.size());
    for (int k = 0; k < f.size(); ++k) {
        Mat col = faces.col(k);
        float x_mean = 0;
        float y_mean = 0;
        for (int i = 0; i < files.size(); ++i) {
            x_mean += col.at<float>(2 * i);
            y_mean += col.at<float>(2 * i + 1);
        }
        Point2f fea(x_mean /= files.size(), y_mean /= files.size());
        mean_shape[k] = fea;
    }
    // 保存平均脸
    Mat mat;
    Rect rect = getFeatureRect(mean_shape);
    mat = Mat::ones(400, 400, CV_8UC3);

    resizeFeatures(mean_shape, -rect.x, -rect.y, 1, 1);
    for (int l = 0; l < mean_shape.size(); ++l) {
        Point2f p = mean_shape.at(l);
        circle(mat, p, 2, Scalar(0, 255, 0), -1, 8, 0);
    }
    imshow("平均脸", mat);
    waitKey(0);
    saveFeatures(mean_shape, "../input/mean_shape.pts", "mean_shape");
    cout << "saved mean face" << endl;

}