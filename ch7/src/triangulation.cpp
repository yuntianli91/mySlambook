// C++ headers
#include <iostream>
// third_party geaders
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ---------------------------------------- function declaration ---------------------------------------------------------------- //
void feature_extractor(Mat img1, Mat img2, vector<KeyPoint>& keypoint_1, vector<KeyPoint>& keypoint_2, vector<DMatch>& selected_matches);
void pose_estimation_2d2d(vector<KeyPoint>& keypoint_1, vector<KeyPoint>& keypoint_2, vector<DMatch>& matches,const Mat K,Mat& R, Mat& t);
void triangulation(const vector<KeyPoint>& keypoints_1,const vector<KeyPoint>& keypoints_2,
                    const vector<DMatch> matches,const Mat K ,Mat& R, Mat& t, vector<Point3d>& points);
Point2f pixel2cam(const KeyPoint keypoint, const Mat& K);
// ---------------------------------------- man function ---------------------------------------------------------------------- //
int main(int argc, char** argv){
    // if (argc != 3){
    //     cout << "usage: pose_estimation file1 file2." << endl;
    //     return 1;
    // }

    Mat img1, img2;
    // img1 = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    // img2 = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);

    img1 = imread("../img/1.png", CV_LOAD_IMAGE_UNCHANGED);
    img2 = imread("../img/2.png", CV_LOAD_IMAGE_UNCHANGED);
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    feature_extractor(img1, img2, keypoints_1, keypoints_2, matches);
    cout << "Totally find " << matches.size() << " feature matches." << endl;
    
    Mat R, t;
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches,K, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, K, R, t, points);
    // reproject to first and second frame
    for (int i = 0; i < (int)matches.size(); i++){
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx], K);
        Point2d pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx], K);
        
        Point2d pt1_trans (
                points[i].x / points[i].z,
                points[i].y / points[i].z
                );
        
        cout << "Coordinates of feature " << (i+1) << " in first frame is: " << pt1_cam << endl
        << "Coordinates of reprojected feature" << (i+1) << " in first frame is: " << pt1_trans << endl
        << "Depth of feature is: " << points[i].z << endl;

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
    
        cout << "Coordinates of feature " << (i+1) << " in second frame is: " << pt2_cam << endl
        << "Coordinates of reprojected feature" << (i+1) << " in second frame is: " << pt2_trans << endl
        << "Depth of feature is: " << points[i].z << endl
        << "--------------------------------------------------------------------------------" << endl;
    }
          
    return 0;
}

// ----------------------------------------- function definations ------------------------------------------------------------- //
void feature_extractor(Mat img1, Mat img2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch>& selected_matches){
    // using orb detect feature and compute descriptor 
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    
    Mat descriptor_1, descriptor_2;
    orb->detectAndCompute(img1, Mat(), keypoints_1, descriptor_1);
    orb->detectAndCompute(img2, Mat(), keypoints_2, descriptor_2);
    // using Flann to find match descriptors
    // FlannBasedMatcher matcher;
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptor_1, descriptor_2, matches);
    // find matchers less than 2 * min_dist or specific distance
    double min_dist = 10000, max_dist = 0;

    for(int i = 0; i < descriptor_1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist){min_dist = dist;}
        if(dist > max_dist){max_dist = dist;}
    }
    
    std::cout << "Minimum distance of descriptors is: " << min_dist << endl;
    std::cout << "Maximum distance of descriptors is: " << max_dist << endl;
    for(int i = 0; i < descriptor_1.rows; i++){
        if (matches[i].distance <= max(2 * min_dist, 30.0) ){
            selected_matches.push_back(matches[i]);
        }
    }
}

void pose_estimation_2d2d(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch>& matches,const Mat K ,Mat& R, Mat& t){
    // camera matrix
    // Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // convert keypoints from vector<KeyPoint> to vector<Point2f>
    vector<Point2f> points1, points2;

    for (int i = 0; i < (int)matches.size(); i++ ){
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    Mat E;
    Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));
    int focal = K.at<double>(1, 1);
    E = findEssentialMat(points1, points2, focal, pp, RANSAC);
    cout << "EssentialMat is :\n" << E << endl;

    recoverPose(E, points1, points2, R, t, focal, pp);
    cout << "Recover R from Essential Mat is :\n" << R << endl;
    cout << "Recover t from Essential Mat is :\n" << t << endl;
}
void triangulation(const vector<KeyPoint>& keypoints_1,const vector<KeyPoint>& keypoints_2,
                    const vector<DMatch> matches,const Mat K ,Mat& R, Mat& t, vector<Point3d>& points)
{
    // transformation matrix of first frame
    Mat T1 = (Mat_<double>(3, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    // transformation matrix of second frame
    Mat T2 = (Mat_<double>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    cout << "T1: " << endl << T1 << endl << "T2: " << endl << T2 << endl;
    // Camera matrix
    // Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521., 249.7, 0, 0, 1);
    // convert from pixel frame to cam frame
    // notice the usage of for iterator
    vector<Point2f> pts_1, pts_2;
    for (DMatch m:matches){
        pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx], K));
        pts_2.push_back(pixel2cam(keypoints_2[m.trainIdx], K));
    }

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
    // convert from homogeneous coordinates to unhomogeneous coordinates
    for (int i=0; i < pts_4d.cols; i++){
        double d = pts_4d.at<float>(3 ,i);
        Point3d p(pts_4d.at<float>(0, i)/d, pts_4d.at<float>(1, i)/d, pts_4d.at<float>(2, i)/d);
        points.push_back(p);
    }
}

Point2f pixel2cam(const KeyPoint keypoint, const Mat& K){
    // convert pixel coordinate of keypoint to Nnormalized camera coordinate
    Point2f pt_pix, pt_cam;

    pt_pix = keypoint.pt;
    pt_cam.x = (pt_pix.x - K.at<double>(0, 2)) / K.at<double>(0, 0); 
    pt_cam.y = (pt_pix.y - K.at<double>(1, 2)) / K.at<double>(1, 1);

    return pt_cam; 
}

