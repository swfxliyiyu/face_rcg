//
// Created by swfxliyiyu on 17-5-8.
//

#include <opencv2/imgcodecs.hpp>
#include "detect_face.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;
using namespace cv;

vector<Rect> detectFace(Mat& img, CascadeClassifier& cascade, double scale = 1.0) {
    // 检测时间
    double t = 0;
    // 存储检测到的脸部方框
    vector<Rect> faces, result_faces;

    Mat gray, smallImg;

    // 将原图转换为灰度图
    cvtColor( img, gray, COLOR_BGR2GRAY );
    // 缩放规模
    double fx = 1 / scale;
    // 缩放图像
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    // 增强图片对比度
    equalizeHist( smallImg, smallImg );
    // 计算时钟
    t = (double)getTickCount();
    // 从缩放后的灰度图中检测人脸
    cascade.detectMultiScale( smallImg, faces,
                              1.1, 2, 0
                                      //|CASCADE_FIND_BIGGEST_OBJECT
                                      //|CASCADE_DO_ROUGH_SEARCH
                                      |CASCADE_SCALE_IMAGE,
                              Size(30, 30));

    t = (double)getTickCount() - t;
    // 输出检测时间
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    // 迭代检测到的脸
    for ( size_t i = 0; i < faces.size(); i++ ) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;

        Rect rect;

        int radius;

        double aspect_ratio = (double)r.width/r.height;
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

