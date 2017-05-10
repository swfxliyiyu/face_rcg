//
// Created by swfxliyiyu on 17-5-8.
//

#ifndef FACE_RCG_DETECT_FACE_H
#define FACE_RCG_DETECT_FACE_H

#include <opencv2/imgcodecs.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"


std::vector<cv::Rect> detectFace(cv::Mat& img, cv::CascadeClassifier& cascade, double scale);


#endif //FACE_RCG_DETECT_FACE_H
