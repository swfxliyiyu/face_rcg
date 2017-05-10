//
// Created by swfxliyiyu on 17-5-8.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include "load_data.h"


std::vector<cv::Point2f> load_features(std::string feature_url) {
    std::vector<cv::Point2f> points;
    std::ifstream input_stream;
    input_stream.open(feature_url);
    // 读取前三行
    for (int i = 0; i < 3; ++i) {
        std::string s;
        getline(input_stream, s);
    }
    // 读取特征点
    for (int i = 0; i < 68; ++i) {
        cv::Point2f point;
        input_stream >> point.x >>point.y;
        points.push_back(point);
    }
    return points;
}


