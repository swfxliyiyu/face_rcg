//
// Created by swfxliyiyu on 17-5-9.
//

#ifndef FACE_RCG_PRE_TREAT_H
#define FACE_RCG_PRE_TREAT_H

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

void resizeAndSave(Mat& src_img, const vector<Point2f>& features, string img_name, int t_w, int t_h);

#endif //FACE_RCG_PRE_TREAT_H
