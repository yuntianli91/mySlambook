// C++ standard headers
#include <iostream>
// third party headers
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
// ----------------------------- Function Declarations -------------------------------------------------------- //
// detect feature and find matches
void feature_extractor(Mat img1, Mat img2, vector<KeyPoint>& keypoint_1, vector<KeyPoint>& keypoint_2, vector<DMatch>& selected_matches);
void pose_estimation_2d2d(vector<KeyPoint>& keypoint_1, vector<KeyPoint>& keypoint_2, vector<DMatch>& matches, Mat& R, Mat& t);
Point2d pixel2cam ( const Point2d& p, const Mat& K );
// ----------------------------- Main function ---------------------------------------------------------------- //
int main(int argc, char** argv){
    if (argc != 3){
        cout << "usage: pose_estimation file1 file2." << endl;
        return 1;
    }

    Mat img1, img2;
    img1 = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    img2 = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    feature_extractor(img1, img2, keypoints_1, keypoints_2, matches);
    cout << "Totally find " << matches.size() << " feature matches." << endl;
    
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
    
    //-- E=t^R*scale
    Mat t_x = ( Mat_<double> ( 3,3 ) <<
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1,0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"t^R="<<endl<<t_x*R<<endl;

    //-- polar strain
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    for ( DMatch m: matches )
    {
        Point2d pt1 = pixel2cam ( keypoints_1[ m.queryIdx ].pt, K );
        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    
    return 0;
}
// ----------------------------- Function Definitions --------------------------------------------------------- //
void feature_extractor(Mat img1, Mat img2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch>& selected_matches){
    // using orb detect feature and compute descriptor 
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    
    Mat descriptor_1, descriptor_2;
    orb->detectAndCompute(img1, Mat(), keypoints_1, descriptor_1);
    orb->detectAndCompute(img2, Mat(), keypoints_2, descriptor_2);
    // using Flann to find match descriptors
    FlannBasedMatcher matcher;
    // BFMatcher matcher(NORM_HAMMING);
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

void pose_estimation_2d2d(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch>& matches, Mat& R, Mat& t){
    // camera matrix
    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // convert keypoints from vector<KeyPoint> to vector<Point2f>
    vector<Point2f> points1, points2;

    for (int i = 0; i < (int)matches.size(); i++ ){
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    Mat F, E, H;
    Point2d pp(325.1, 249.7);
    int focal = 521;
    F = findFundamentalMat(points1, points2, CV_FM_8POINT);
    // F = findFundamentalMat(points1, points2, CV_FM_8POINT, 3.0, 0.98, Mat());
    cout << "FundamentalMat is :\n" << F << endl;
    E = findEssentialMat(points1, points2, focal, pp, RANSAC);
    cout << "EssentialMat is :\n" << E << endl;
    H = findHomography(points1, points2, RANSAC, 3.0, noArray() );
    cout << "HomographyMat is :\n" << H << endl;

    recoverPose(E, points1, points2, R, t, focal, pp);
    cout << "Recover R from Essential Mat is :\n" << R << endl;
    cout << "Recover t from Essential Mat is :\n" << t << endl;
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

