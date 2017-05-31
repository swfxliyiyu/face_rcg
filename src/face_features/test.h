//
// Created by yiyuli@pku.edu.cn on 17-5-31.
//

#ifndef FACE_RCG_TEST_CPP_H
#define FACE_RCG_TEST_CPP_H
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <opencv/cv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "map"
#include "test.h"
#include "load_data.h"
#include "detect_face.h"

using namespace std;
using namespace cv;
void resizeTestImg(Mat &src_img, vector<Point2f> &features, Rect face_rect, string img_name, int t_w, int t_h);


#endif //FACE_RCG_TEST_CPP_H
