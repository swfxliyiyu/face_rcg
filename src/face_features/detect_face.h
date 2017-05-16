//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

#ifndef FACE_RCG_DETECT_FACE_H
#define FACE_RCG_DETECT_FACE_H

#include <opencv2/imgcodecs.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "map"
using namespace std;
using namespace cv;


std::vector<cv::Rect> detectFaceRect(cv::Mat &img, double scale);
map<string, float> rectMeanStdDev(string pts_path);
void computeSIFT(Mat &src_img, vector<Point2f> &feas, Mat &dst);
vector<Point2f> randomGenerFeartures(const Rect &face_rect,
                                     const map<string, float> &mean_sdev,
                                     const Size &img_size,
                                     const vector<Point2f> &mean_shape);
void trainModel(string input_path, int itr, int n_vk);
void test(Mat &img, string input_path, int itr);
Mat generMatOfFeature(const vector<Point2f> &feas);
Mat computeSIFT(Mat &src_img, vector<Point2f> &feas);

#endif //FACE_RCG_DETECT_FACE_H
