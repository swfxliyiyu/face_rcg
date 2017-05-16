//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

#ifndef FACE_RCG_LOAD_DATA_H
#define FACE_RCG_LOAD_DATA_H
using namespace std;
using namespace cv;

std::vector<cv::Point2f> load_features(std::string);
std::vector<std::string> getFiles(std::string dirname);
Mat loadMat(string mat_url);

#endif //FACE_RCG_LOAD_DATA_H
