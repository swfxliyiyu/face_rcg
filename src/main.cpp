//
// Created by yiyuli@pku.edu.cn on 17-5-9.
//

#include "opencv2/objdetect.hpp"
#include "face_features/pre_treat.h"

using namespace std;
using namespace cv;


string cascadeName = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "/opt/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string imgName = "../input/image_0001.png";



int main() {

//    prt_images();
    string dir_path = "../input/images/";
//    vector<string> files = getFiles(dir_path);
//    for (vector<string>::iterator i = --files.end(); i >= files.begin(); --i) {
//        string::iterator sit = --(i->end());
//        if (*sit != 'g') {
//            files.erase(i);
//        }
//    }
//
//    for (vector<string>::iterator i = files.begin(); i < files.end(); ++i) {
//        cout << *i << endl;
//    }
//    // 对于每个.png文件，提取文件名，处理图片
//    for (vector<string>::iterator i = files.begin(); i < files.end(); ++i) {
//        Mat prt_img = imread(dir_path + *i);
//        cout << dir_path + *i << endl;
//        int cnt = 0;
//        for (string::iterator it = --(i->end()); cnt < 4; ++cnt, --it) {
//            i->erase(it);
//        }
//        vector<Point2f> prt_features = load_features(dir_path + *i + ".pts");
//        for (int j = 0; j < prt_features.size(); ++j) {
//            Point2f feature = prt_features.at(j);
//            circle(prt_img, feature, 5, Scalar(0, 255, 0), -1, 8, 0);
//        }
//        imshow("", prt_img);
//        waitKey(0);
//    }
    getMeanShape(dir_path);


}
