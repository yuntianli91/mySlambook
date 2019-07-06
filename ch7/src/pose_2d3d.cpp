// C++ headers
#include <iostream>
#include <chrono>
//third party headers
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;
using namespace cv;

// ----------------------------------------- function declaration ------------------------------------------------------------- //
void feature_extractor(Mat img1, Mat img2, vector<KeyPoint>& keypoint_1, vector<KeyPoint>& keypoint_2, vector<DMatch>& selected_matches);
Point2f pixel2cam(const KeyPoint keypoint, const Mat& K);
void initiatePNP(const Mat dep_1, const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2, const vector<DMatch>& matches, const Mat& K, 
                vector<Point3d>& pts_3d, vector<Point2d>& pts_2d, Mat& r, Mat& t);
void optimizeWithG2O(const vector<Point3d>& pts_3d, const vector<Point2d>& pts_2d, const Mat& K, const Mat& R0, const Mat&t0, Mat& r, Mat& t);

// ----------------------------------------- main function -------------------------------------------------------------------- // 
int main(int argc, char** argv){
    // color image and depth image
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
    Mat dep_1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    // camera matrix
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // extract features
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    feature_extractor(img_1, img_2, keypoints_1, keypoints_2, matches);
    
    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    Mat r0, t0;
    initiatePNP(dep_1, keypoints_1, keypoints_2, matches, K, pts_3d, pts_2d, r0, t0);

    Mat R0;
    cv::Rodrigues(r0, R0);

    cout << "R = " << R0 << endl;
    cout << "t = " << t0 << endl;

    Mat r, t;
    optimizeWithG2O(pts_3d, pts_2d, K, R0, t0, r, t);

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

Point2f pixel2cam(const KeyPoint keypoint, const Mat& K){
    // convert pixel coordinate of keypoint to Nnormalized camera coordinate
    Point2f pt_pix, pt_cam;

    pt_pix = keypoint.pt;
    pt_cam.x = (pt_pix.x - K.at<double>(0, 2)) / K.at<double>(0, 0); 
    pt_cam.y = (pt_pix.y - K.at<double>(1, 2)) / K.at<double>(1, 1);

    return pt_cam; 
}

void initiatePNP(const Mat dep_1, const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2, const vector<DMatch>& matches, const Mat& K, 
                vector<Point3d>& pts_3d, vector<Point2d>& pts_2d, Mat& r, Mat& t)
{
    // construct 3D and 2D points
    int index_col, index_row;
    for (DMatch m:matches){
        index_col = (int)keypoints_1[m.queryIdx].pt.x;
        index_row = (int)keypoints_1[m.queryIdx].pt.y;
        ushort depth = dep_1.ptr<ushort>(index_row)[index_col];

        if (depth == 0){continue;}
        // conver from mm to m   
        float dd = depth/1000.0;
        Point2d p_cam = pixel2cam(keypoints_1[m.queryIdx], K);
        pts_3d.push_back(Point3d(p_cam.x * dd, p_cam.y * dd, dd)); 
    
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    solvePnP(pts_3d, pts_2d, K, noArray(), r, t, false, cv::SOLVEPNP_EPNP);
}

void optimizeWithG2O(const vector<Point3d>& pts_3d, const vector<Point2d>& pts_2d, const Mat& K, const Mat& R0, const Mat&t0, Mat& r, Mat& t){
    // initiate sparse optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
    Block::LinearSolverType* linear_solver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linear_solver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    // config camera parameters
    // g2o::CameraParameters(double focal_length, Vector2d p_point, double baseline)
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(1, 1), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // add vertex(VertexSE3Expmap(), VertexSBAPointXYZ)
    // pose vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    Eigen::Matrix3d R_mat;
    R_mat << 
        R0.at<double>(0, 0), R0.at<double>(0, 1), R0.at<double>(0, 2),
        R0.at<double>(1, 0), R0.at<double>(1, 1), R0.at<double>(1, 2),
        R0.at<double>(2, 0), R0.at<double>(2, 1), R0.at<double>(2, 2);
    g2o::SE3Quat quat_0(R_mat, Eigen::Vector3d(t0.at<double>(0, 0), t0.at<double>(1, 0), t0.at<double>(2, 0)));
    pose->setEstimate(quat_0);
    optimizer.addVertex(pose);
        // landmark vertex
    int index = 1;
    for (Point3d pt3:pts_3d){
        g2o::VertexSBAPointXYZ* landmark = new g2o::VertexSBAPointXYZ();
        landmark->setId(index++);
        landmark->setEstimate(Eigen::Vector3d(pt3.x, pt3.y, pt3.z));
        // marginalize landmarks as g2o only accept poseMatrixType;
        landmark->setMarginalized(true);
        optimizer.addVertex(landmark);
    }
    // add edge(EdgeProjectXYZ2UV)
    index = 1;
    for (Point2d pt2:pts_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        // add vertex, using dynamic_cast to make sure the right pointer type
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        // edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setVertex(1, pose);
        edge->setId(index++);
        edge->setMeasurement(Eigen::Vector2d(pt2.x, pt2.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());

        optimizer.addEdge(edge);
    }
    optimizer.setVerbose(true);

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_cost = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "optimization time cost " << time_cost.count() << " seconds." << endl;
    // construct Isometry3d with SE3Quat then call matrix() to print transformation matrix.
    cout << "After optimization, T is :" << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
   }
 