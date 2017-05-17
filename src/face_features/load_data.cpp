//
// Created by yiyuli@pku.edu.cn on 17-5-8.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include "load_data.h"
#include "pre_treat.h"
#include <dirent.h>
#include "opencv2/contrib/contrib.hpp"
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>


using namespace std;
using namespace cv;

vector<string> getFiles(string dirname) {
    DIR *dp;
    struct dirent *dirp;

    // 存储文件名
    vector<string> file_names;

    if ((dp = opendir(dirname.c_str())) == NULL) {
        perror("opendir error");
        exit(1);
    }

    while ((dirp = readdir(dp)) != NULL) {
        if ((strcmp(dirp->d_name, ".") == 0) || (strcmp(dirp->d_name, "..") == 0))
            continue;
        file_names.push_back(string(dirp->d_name));
        //printf("%6d:%-19s %5s\n",dirp->d_ino,dirp->d_name,(dirp->d_type==DT_DIR)?("(DIR)"):(""));
    }

    return file_names;
}

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

/**
 * 根据指定地址载入矩阵
 * @param mat_url
 * @return
 */
Mat loadMat(string mat_url) {
    ifstream input(mat_url);
    int rows, cols;
    input >> rows;
    input >> cols;
    string temp;
    getline(input, temp);

    Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        string line;
        getline(input, line);
        vector<string> tokens;
        boost::split(tokens, line, boost::is_any_of("[, ]"));
        int index = 0;
        for (int j = 0; j < cols; ++j) {
            while (tokens[index].compare("") == 0)  {
                index++;
            }
            float t;
            sscanf(tokens[index++].c_str(), "%8f", &t);
            mat.at<float>(i, j) = t;
        }
    }
    cout << mat_url << ":" << mat.size << endl;
    return mat;
}

bool existFile(string path) {
    // 判断文件是否存在

    if (access(path.c_str(), 0) != -1) {
        cout << path << "已存在" << endl;
        return true;
    }
    return false;
}