//
// Created by yiyuli@pku.edu.cn on 17-5-9.
//

#ifndef FACE_RCG_PRE_TREAT_H
#define FACE_RCG_PRE_TREAT_H

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

void resizeAndSave(Mat& src_img, vector<Point2f>& features, string img_name, int t_w, int t_h);
void prt_images();
vector<string> getFiles(string dirname);
void getMeanShape(string dirpath);
vector<string> getImgNames(string dir_path);
Rect getFeatureRect(const vector<Point2f> &vector);
void resizeFeatures(vector<Point2f> &features, int x_move, int y_move, float w_scale, float h_scale);

#endif //FACE_RCG_PRE_TREAT_H
