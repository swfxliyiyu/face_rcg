#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "../src/face_features/load_data.h"
#include "../src/face_features/detect_face.h"
#include "face_features/pre_treat.h"

using namespace std;
using namespace cv;


string cascadeName = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "/opt/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string imgName = "../input/image_0001.png";

//int main( int argc, const char** argv ) {
//    Mat image;
//    bool tryflip = true;
//    CascadeClassifier cascade, nestedCascade;
//    double scale;
//
//    cv::CommandLineParser parser(argc, argv,
//                                 "{help h||}"
//                                         "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
//                                         "{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
//                                         "{scale|1|}{try-flip||}{@filename||}"
//    );
//
//    scale = 1;
//    if (scale < 1)
//        scale = 1;
//
//    if ( !nestedCascade.load( nestedCascadeName ) )
//        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
//
//    if( !cascade.load( cascadeName ) ) {
//        cerr << "ERROR: Could not load classifier cascade" << endl;
//        help();
//        return -1;
//    } else {
//        image = imread(imgName, CV_LOAD_IMAGE_COLOR );
//        if(image.empty()) cout << "Couldn't read " + imgName << endl;
//    }
//
//
//
//
//
//    cout << "Detecting face(s) in " << imgName << endl;
//    if( !image.empty() )
//    {
//        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
//        waitKey(0);
//    }
//    else if( !image.empty() )
//    {
//        /* assume it is a text file containing the
//        list of the image filenames to be processed - one per line */
//        FILE* f = fopen( imgName.c_str(), "rt" );
//        if( f )
//        {
//            char buf[1000+1];
//            while( fgets( buf, 1000, f ) )
//            {
//                int len = (int)strlen(buf);
//                while( len > 0 && isspace(buf[len-1]) )
//                    len--;
//                buf[len] = '\0';
//                cout << "file " << buf << endl;
//                image = imread( buf, 1 );
//                if( !image.empty() )
//                {
//                    detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
//                    char c = (char)waitKey(0);
//                    if( c == 27 || c == 'q' || c == 'Q' )
//                        break;
//                }
//                else
//                {
//                    cerr << "Aw snap, couldn't read image " << buf << endl;
//                }
//            }
//            fclose(f);
//        }
//    }
//
//
//    return 0;
//}

int main() {
//    Mat img = imread("../input/image_0001.png");
//    vector<Point2f> points = load_features("../input/image_0001.pts");
//    for (int i = 0; i < points.size(); ++i) {
//        circle(img, points[i], 3, Scalar(0, 0, 255), CV_FILLED, CV_AA);
//    }
//    imshow("", img);
//    waitKey();
//    return 0;
    Mat img = imread("../input/image_0001.png");
    vector<Point2f> features = load_features("../input/image_0001.pts");
    resizeAndSave(img, features, "image_0001", 400, 400);
//    CascadeClassifier cascade;
//    cascade.load("/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
//    vector<Rect> faces = detectFace(img, cascade, 1);
//    for (int i = 0; i < faces.size(); ++i) {
//        Rect rect = faces.at(i);
//        cout << rect.x << " " << rect.y;
//        rectangle(img, rect, Scalar(255, 0, 0));
//    }
//    imshow("", img);
//    waitKey(0);
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
            {
                    Scalar(255,0,0),
                    Scalar(255,128,0),
                    Scalar(255,255,0),
                    Scalar(0,255,0),
                    Scalar(0,128,255),
                    Scalar(0,255,255),
                    Scalar(0,0,255),
                    Scalar(255,0,255)
            };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
                              1.1, 2, 0
                                      //|CASCADE_FIND_BIGGEST_OBJECT
                                      //|CASCADE_DO_ROUGH_SEARCH
                                      |CASCADE_SCALE_IMAGE,
                              Size(30, 30));
    if( tryflip )
    {
        (smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                  1.1, 2, 0
                                          //|CASCADE_FIND_BIGGEST_OBJECT
                                          //|CASCADE_DO_ROUGH_SEARCH
                                          |CASCADE_SCALE_IMAGE,
                                  Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
                                        1.1, 2, 0
                                                //|CASCADE_FIND_BIGGEST_OBJECT
                                                //|CASCADE_DO_ROUGH_SEARCH
                                                //|CASCADE_DO_CANNY_PRUNING
                                                |CASCADE_SCALE_IMAGE,
                                        Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );

        }
    }
    imshow( "result", img );
    cout << "wight: " << img.cols << ", height: " << img.rows;
}