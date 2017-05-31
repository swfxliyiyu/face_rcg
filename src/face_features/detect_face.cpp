//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

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

/**
 * 检测人脸方框
 * @param img 输入图片
 * @param scale 图片放大规模
 * @return 检测到的人脸框
 */
vector<Rect> detectFaceRect(const Mat &img, double scale = 1.0) {
    // 分类器
    CascadeClassifier cascade;
    cascade.load("/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
    // 检测时间
    double t = 0;
    // 存储检测到的脸部方框
    vector<Rect> faces, result_faces;

    Mat gray, smallImg;

    // 将原图转换为灰度图
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // 缩放规模
    double fx = 1 / scale;
    // 缩放图像
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    // 增强图片对比度
    equalizeHist(smallImg, smallImg);
    // 计算时钟
    t = (double) getTickCount();
    // 从缩放后的灰度图中检测人脸
    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0
                                     //|CASCADE_FIND_BIGGEST_OBJECT
                                     //|CASCADE_DO_ROUGH_SEARCH
                                     | CASCADE_SCALE_IMAGE,
                             Size(30, 30));

    t = (double) getTickCount() - t;
    // 输出检测时间
//    printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
    // 迭代检测到的脸
    for (size_t i = 0; i < faces.size(); i++) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;

        Rect rect;

//        int radius;
//        double aspect_ratio = (double) r.width / r.height;
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
 * 获取特征框和人脸检测框的均差和标准差
 * return int[x_mean, y_mean, w_mean, h_mean, x_var, y_var, w_var, h_var]
 */
map<string, float> rectMeanStdDev(string pts_path) {


    map<string, float> mean_var;
    vector<string> file_names = getImgNames(pts_path);
    int file_count = (int) file_names.size();
    // 用于存储每对方框的差
    Mat x_diff(0, 0, CV_32FC1),
            y_diff(0, 0, CV_32FC1),
            w_diff(0, 0, CV_32FC1), h_diff(0, 0, CV_32FC1);

    for (int i = 0; i < file_count; ++i) {
        string file_name = file_names[i];
        vector<Point2f> features = load_features(pts_path + file_name + ".pts");
        Rect featureRect = getFeatureRect(features);
        Mat img = imread(pts_path + file_name + ".jpg");
        cout << file_name << endl;
        vector<Rect> face_rects = detectFaceRect(img, 2.0);
        // TODO
        if (face_rects.empty()) {
            cout << "图片" << file_name << "检测不到人脸!" << endl;
            continue;
        }
        // 取与特征点最近的一张人脸框
        Rect face_rect;
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
        x_diff.push_back(featureRect.x - face_rect.x);
        y_diff.push_back(featureRect.y - face_rect.y);
        w_diff.push_back(featureRect.width - face_rect.width);
        h_diff.push_back(featureRect.height - face_rect.height);
    }
    // x, y ,w, h 的方差和标准差
    Scalar x_mean, x_sdev, y_mean, y_sdev, w_mean, w_sdev, h_mean, h_sdev;
    meanStdDev(x_diff, x_mean, x_sdev);
    meanStdDev(y_diff, y_mean, y_sdev);
    meanStdDev(w_diff, w_mean, w_sdev);
    meanStdDev(h_diff, h_mean, h_sdev);
    mean_var["x_mean"] = (float) x_mean[0];
    mean_var["x_sdev"] = (float) x_sdev[0];
    mean_var["y_mean"] = (float) y_mean[0];
    mean_var["y_sdev"] = (float) y_sdev[0];
    mean_var["w_mean"] = (float) w_mean[0];
    mean_var["w_sdev"] = (float) w_sdev[0];
    mean_var["h_mean"] = (float) h_mean[0];
    mean_var["h_sdev"] = (float) h_sdev[0];
    // 保存结果
    ofstream output("../input/mean_sdev.txt");
    for (map<string, float>::iterator it = mean_var.begin(); it != mean_var.end(); ++it) {
        output << it->first << " " << it->second << endl;
    }
    output.close();

    return mean_var;
}

/**
 * 计算给定图片和特征点的SIFT特征描述
 * @param src_img
 * @param feas
 * @param dst 128n*1维SIFT特征描述
 */
Mat computeSIFT(Mat &src_img, vector<Point2f> &feas) {

//    cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create(100);
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(0);
    // 描述子
    Mat descriptors;
    for (int i = 0; i < feas.size(); ++i) {
        // 用掩码确定位置
        Point2f p = feas[i];
        Mat mask = Mat::zeros(src_img.rows, src_img.cols, CV_8U);
        for (int j = (int) (round(p.x) - 16); j < round(p.x) + 16; ++j) {
            for (int k = (int) (round(p.y) - 16); k < (int) (round(p.y) + 16); ++k) {
                mask.at<char>(Point(j, k)) = 1;
            }
        }
//        cout << mask << endl;
        Mat des_temp;
        vector<KeyPoint> kps;
        f2d->detect(src_img, kps, mask);
        // 排序关键点
        sort(kps.begin(), kps.end(),
             [&](KeyPoint a, KeyPoint b){ return (bool) (a.response > b.response);});
        cout << kps.size() << "个关键点" << endl;
        int k = kps.size();
        if (kps.size() > 1) {
            kps.resize(1);
        }
        f2d->compute(src_img, kps, des_temp);
//        cout << kps.size() << "个关键点可检测" << endl;
        int r = des_temp.rows;
        for (int l = 0; l < 1 - r; ++l) {
            des_temp.push_back(Mat::zeros(1, 128, CV_32F));
        }
        for (int m = 0; m < 1; ++m) {
            descriptors.push_back(des_temp.row(m));
        }

    }
    cout << descriptors.size<< endl;
    return descriptors.reshape(0, descriptors.rows * descriptors.cols);
}

/**
 * 计算给定图片和特征点的SIFT特征描述
 * @param src_img
 * @param feas
 * @param dst 128n*1维SIFT特征描述
 */
Mat computeSIFT(Mat &src_img, Mat &feas) {
    vector<Point2f> points;
    for (int i = 0; i < feas.rows / 2; ++i) {
        Point2f p;
        p.x = feas.at<float>(2 * i);
        p.y = feas.at<float>(2 * i + 1);
        points.push_back(p);
    }

    return computeSIFT(src_img, points);
}

/**
 * 根据特征点生成矩阵
 * @param feas
 * @return
 */
Mat generMatOfFeature(const vector<Point2f> &feas) {
    Mat mat((int) (feas.size() * 2), 1, CV_32F);
    for (int i = 0; i < feas.size(); ++i) {
        Point2f p = feas[i];
        mat.at<float>(2 * i) = p.x;
        mat.at<float>(2 * i + 1) = p.y;
    }
    return mat;
}


/**
 * 训练模型
 * @param input_path
 * @param itr
 * @param n_vk
 */
void trainModel(string input_path, int itr, int n_vk) {

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
    set<string> name_set(img_names.begin(), img_names.end());
    int i = 0;
    for (set<string>::iterator iterator1 = name_set.begin(); i < img_names.size(); ++i, ++iterator1) {
        // 获取图片名
        string img_name = *iterator1;
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
    // 对fk进行pca压缩
//    PCA pca(pca_fk_mat, Mat(), CV_PCA_DATA_AS_ROW, 0);
    for (int l = 0; l < pca_fk_mat.rows;) {
        for (map<string, ImgInfo>::iterator it = img_map.begin();
             it != img_map.end(); ++it) {
            ImgInfo &info = it->second;
            for (int i = 0; i < n_vk; ++i) {
                Mat fk_t = pca_fk_mat.row(l++);
//                Mat pca_fk(0, 0, CV_32F);
//                pca_fk = pca.project(fk_t);
//                fk_t.copyTo(pca_fk);
//                cout << pca_fk << endl;
//                cout << pca_fk.size << endl;
//                Mat fk = pca_fk.t();
                Mat fk = fk_t.t();
                fk.push_back(1.f);
                info.vector_fk.push_back(fk);
            }
        }
    }


    // 生成R矩阵（增广）
    Mat R_k(0, 0, CV_32F);
    // 迭代求解矩阵模型
    for (int i = 0; i < itr; ++i) {
        cout << "正在进行第 " << i + 1 << " 次迭代..." << endl;
        if (i != 0) {
            // 用于保存fk
            pca_fk_mat = Mat(0, 0, CV_32F);
            cout << "正在生成本轮的vk,dk,fk ..." << endl;
            // 生成本轮vk，dk，fk
            for (map<string, ImgInfo>::iterator it = img_map.begin();
                 it != img_map.end(); ++it) {
                string name = it->first;
                ImgInfo &info = it->second;
                Mat fea = info.fea;
                for (int j = 0; j < info.vector_vk.size(); ++j) {
                    Mat img = imread(input_path + "images/" + name + ".jpg");
                    Mat &vk = info.vector_vk[j];
                    Mat &fk = info.vector_fk[j];
                    vk = vk + (R_k * fk);
                    info.vector_dk[j] = (fea - (vk));
                    cout << "计算图片" << name << "的SIFT" << endl;
                    Mat fk_mat = computeSIFT(img, vk);
                    // 将新的fk放入pca_fk_mat中
                    pca_fk_mat.push_back(fk_mat.t());
//                    fk_mat.push_back(1.f);
//                    fk = fk_mat;

                }
            }
            // 重新生成pca
//            pca = PCA(pca_fk_mat, Mat(), CV_PCA_DATA_AS_ROW, 0);
            // 计算将fk进行pca变换
            int pca_row = 0;
            for (map<string, ImgInfo>::iterator it = img_map.begin();
                 it != img_map.end(); ++it) {
                string name = it->first;
                ImgInfo &info = it->second;
                // 将本轮的fk清空
                info.vector_fk.clear();
                // 每张图放入n_vk个fk
                for (int j = 0; j < n_vk; ++j) {
                    Mat fk_t = pca_fk_mat.row(pca_row++);
//                    Mat pca_fk(0, 0, CV_32F);
//                    pca_fk = pca.project(fk_t);
//                    fk_t.copyTo(pca_fk);
//                    cout << pca_fk << endl;
//                    cout << pca_fk.size << endl;
//                    Mat fk = pca_fk.t();
                    Mat fk = fk_t.t();
                    fk.push_back(1.f);
                    info.vector_fk.push_back(fk);
                }
            }
        }

        // 保存pca矩阵
        // TODO
//        ostringstream ss;
//        ss << "../output/model/p" << i << ".pca";
//        ofstream os(ss.str());
//        os << pca.eigenvectors.rows << " " << pca.eigenvectors.cols << endl;
//        os << pca.eigenvectors;
//        os.close();
//        ostringstream ss1;
//        ss1 << "../output/model/p" << i << ".m";
//        os.open(ss1.str());
//        os << pca.mean.rows << " " << pca.mean.cols << endl;
//        os << pca.mean;
//        os.close();

        // R_kj = (x_t*x)-1 * x_t * y
        // 计算本轮的x
        Mat x(0, 0, CV_32F);
        for (map<string, ImgInfo>::iterator it = img_map.begin();
             it != img_map.end(); ++it) {
            ImgInfo &info = it->second;
            for (int k = 0; k < info.vector_fk.size(); ++k) {
                x.push_back(info.vector_fk[k].t());
            }
        }
        // 计算本轮的R_k
        cout << "正在计算本轮的R_k..." << endl;
        R_k = Mat(0, 0, CV_32F);

        Mat y(0, 0, CV_32F);
        for (map<string, ImgInfo>::iterator it = img_map.begin();
             it != img_map.end(); ++it) {
            ImgInfo info = it->second;
            for (int k = 0; k < info.vector_dk.size(); ++k) {
                y.push_back(info.vector_dk[k].t());
            }
        }

        Mat x_t = x.t();
        cout << "正在计算R_k" << endl;
//            R_k = (x_t*x).inv()*(x_t*y);

        solve(x_t*x, x_t*y, R_k, DECOMP_SVD);
        R_k = R_k.t();
        cout << R_k << endl;


        // 保存R_k

        cout << "正在保存R_" << i << "..." << endl;
        ostringstream oss;
        oss << "../output/model/R" << i << ".mdl";
        ofstream output;
        output.open(oss.str());
        output << R_k.rows << " " << R_k.cols << endl;
        output << R_k;
        output.close();

//        for (map<string, ImgInfo>::iterator it = img_map.begin();
//             it != img_map.end(); ++it) {
//            ImgInfo &img = it->second;
//            string imgname = it->first;
//            Mat v0 = img.vector_vk[0];
//            Mat image = imread("../input/images/" + imgname + ".jpg");
//            Mat f0 = img.vector_vk[0];
//
//            Mat R = R_k;
//            pca.project(f0.t(), f0);
//
//            f0 = f0.t();
////            cout << f0.t() << endl;
//
//            f0.push_back(1.f);
//            Mat d0 = (R*f0);
//            Mat v1 = v0 + d0;
//
////            cout << "Rk" << R << endl;
////            cout << "fk" << f0<< endl;
////            cout << "d0" << d0 << endl;
////            cout << "v1" << v0 << endl;
//
//            for (int l = 0; l < v0.rows / 2; ++l) {
//                Point2f p1(v1.at<float>(2*l), v1.at<float>(2*l+1));
//
//                circle(image, p1, 2, Scalar(0,255,255), -1,8,0);
//                cout << p1 << endl;
//            }
//            imshow("", image);
//            waitKey(0);
//        }
    }
}


/**
 * 随机生成高斯分布整数
 * @param mean
 * @param sdev
 * @return
 */
int seed = 0;

int normlRandom(float mean, float sdev) {
    default_random_engine engine((unsigned long) (time(0) + (++seed)));
    normal_distribution<float> distribution(mean, (float) (sdev / 2.333));
    return (int) lround(distribution(engine));
}

/**
 * 根据人脸框和平均脸随机生成相应特征点
 * @param face_rect
 * @param mean_sdev
 * @param img_size
 * @param mean_shape
 * @return
 */
vector<Point2f> randomGenerFeartures(const Rect &face_rect,
                                     const map<string, float> &mean_sdev,
                                     const Size &img_size,
                                     const vector<Point2f> &mean_shape) {
    int x, y, w, h;
    do {
        // 根据均值标准差，随机生成特征点方框

        int x_diff = normlRandom(mean_sdev.at("x_mean"), mean_sdev.at("x_sdev"));
        int y_diff = normlRandom(mean_sdev.at("y_mean"), mean_sdev.at("y_sdev"));
        int w_diff = normlRandom(mean_sdev.at("w_mean"), mean_sdev.at("w_sdev"));
        int h_diff = normlRandom(mean_sdev.at("h_mean"), mean_sdev.at("h_sdev"));

        x = face_rect.x + x_diff;
        y = face_rect.y + y_diff;
        w = face_rect.width + w_diff;
        h = face_rect.height + h_diff;
    } while (x < 0 || y < 0 || x + w >= img_size.width || y + h >= img_size.height || w <= 0 || h <= 0);

    Rect mean_rect = getFeatureRect(mean_shape);
    vector<Point2f> fea(mean_shape);
    resizeFeatures(fea, x - mean_rect.x, y - mean_rect.y,
                   (float) w / mean_rect.width, (float) h / mean_rect.height);
    return fea;

}


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