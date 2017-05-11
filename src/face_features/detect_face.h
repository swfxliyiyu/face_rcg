//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

#ifndef FACE_RCG_DETECT_FACE_H
#define FACE_RCG_DETECT_FACE_H

#include <opencv2/imgcodecs.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"


std::vector<cv::Rect> detectFaceRect(cv::Mat &img, double scale);


#endif //FACE_RCG_DETECT_FACE_H
