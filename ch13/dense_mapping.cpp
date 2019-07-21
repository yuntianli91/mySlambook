/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-21 15:05:51
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-21 17:08:48
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace Sophus;

// ====================== parameters declarations============================ //
const int boarder = 20; // image boarder in pixels
const int width = 640; // image width in pixels
const int height = 480; // image height in pixels

const double fx = 481.2; // camera intrinsic
const double fy = -480.0; // camera intrinsic
const double cx = 319.5; // camera intrinsic
const double cy = 239.5; // camera intrinsic

const int ncc_window_size = 2; // half size of NCC window 
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // add 1 to make odd area 

const double min_cov = 0.1; // minimum convariance
const double max_cov = 10; // maximum convariance
// ====================== function declarations =============================== //
bool readDataset(const string& filepath, vector<string>& image_files, vector<SE3d>& poses); // read dataset file
bool updateDepth(const Mat& ref, const Mat& curr, const SE3d& T_C_R, Mat& depth, Mat& depth_cov); // update depth estimation
bool epipolarSearch(const Mat& ref, const Mat& curr, const SE3d& T_C_R, const Point2f& pt_ref, 
    const double& depth_mu, const double& depth_cov, Point2f& pt_curr);// search point on epipolar line
bool updateDepthFilter(const Point2f& pt_ref, const Point2f& pt_curr, const SE3d& T_C_R, Mat& depth, Mat& depth_cov);
double NCC(const Mat& ref, const Mat& curr, const Point2f pt_ref, const Point2f pt_curr);// ncc block matching
// linear interpolation
inline double getBilinearInterpolatedValue( const Mat& img, const Point2f& pt ) {
    int x = pt.x; 
    int y = pt.y;
    double dx = x - floor(x); 
    double dy = y - floor(y);

    const uchar* d_ptr = img.ptr<uchar>(x);
    return  (( 1-dx ) * ( 1-dy ) * double(d_ptr[0]) +
            dx* ( 1-dy ) * double(d_ptr[1]) +
            ( 1-dx ) * dy* double(d_ptr[img.step + 0]) +
            dx * dy * double(d_ptr[img.step+1])) / 255.0; // why 255 ?
}
// ------------------------------------------------------------------
// 一些小工具 
// 显示估计的深度图 
void plotDepth( const Mat& depth );

// 像素到相机坐标系 
inline Vector3d px2cam ( const Point2f px ) {
    
    return Vector3d ( 
        (px.x - cx)/fx,
        (px.y - cy)/fy, 
        1
    );
}

// 相机坐标系到像素 
inline Point2f cam2px ( const Vector3d p_cam ) {
    return Point2f (
        p_cam(0,0)*fx/p_cam(2,0) + cx, 
        p_cam(1,0)*fy/p_cam(2,0) + cy 
    );
}

// 检测一个点是否在图像边框内
inline bool inside( const Point2f& pt ) {
    return pt.x >= boarder && pt.y>=boarder 
        && pt.x+boarder<width && pt.y+boarder<=height;
}

// 显示极线匹配 
void showEpipolarMatch( const Mat& ref, const Mat& curr, const Point2f& px_ref, const Point2f& px_curr );

// 显示极线 
void showEpipolarLine( const Mat& ref, const Mat& curr, const Point2f& px_ref, const Point2f& px_min_curr, const Point2f& px_max_curr );
// ====================== main function ======================================= //
int main(int argc, char** argv){

}
// ====================== function definitions ================================ //
// read dataset file
bool readDataset(const string& filepath, vector<string>& image_files, vector<SE3d>& poses){
    ifstream fin(filepath + "/first_200_frames_traj_over_table_input_sequence.txt");
    if ( !fin.is_open() ) return false;

    while(fin.peek()!=EOF){
        string img;
        fin >> img;
        string image_file = filepath + "/images/" + img;
        image_files.push_back(image_file);

        double data[7];
        for(double& d:data) fin >> d; // add & as data wil be modified
        poses.push_back(SE3d(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Eigen::Vector3d(data[0], data[1], data[2])
        ));

        if(!fin.good()) break;
    }
    return true;
}

bool updateDepth(const Mat& ref, const Mat& curr, const SE3d& T_C_R, Mat& depth, Mat& depth_cov){
    // 遍历每个像素
#pragma omp parallel for // for loop will be run in multi parallel threads
    for ( int x=boarder; x<width-boarder; x++ )
#pragma omp parallel for
        for ( int y=boarder; y<height-boarder; y++ )
        {
            if ( depth_cov.ptr<double>(y)[x] < min_cov || depth_cov.ptr<double>(y)[x] > max_cov ) // 深度已收敛或发散
                continue;
            // 在极线上搜索 (x,y) 的匹配 
            Point2f pt_curr; 
            bool ret = epipolarSearch ( 
                ref, 
                curr, 
                T_C_R, 
                Point2f(x,y), 
                depth.ptr<double>(y)[x], 
                sqrt(depth_cov.ptr<double>(y)[x]),
                pt_curr
            );
            
            if ( ret == false ) // 匹配失败
                continue; 
            
            Point2f pt_ref(x,y);
			// 取消该注释以显示匹配
            showEpipolarMatch( ref, curr, pt_ref, pt_curr );
            
            // 匹配成功，更新深度图 
            updateDepthFilter( pt_ref, pt_curr, T_C_R, depth, depth_cov );
        } 
}
// search point on epipolar line

bool epipolarSearch(const Mat& ref, const Mat& curr, const SE3d& T_C_R, const Point2f& pt_ref, 
    const double& depth_mu, const double& depth_cov, Point2f& pt_curr){

    }
    
double NCC(const Mat& ref, const Mat& curr, const Point2f pt_ref, const Point2f pt_curr);// ncc block matching

// show depth image
void plotDepth( const Mat& depth ){
    imshow("depth", depth * 0.4);
    waitKey(1);
}

// show epipolar match results
void showEpipolarMatch( const Mat& ref, const Mat& curr, const Point2f& px_ref, const Point2f& px_curr ){
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle( ref_show, px_ref, 5, cv::Scalar(0,0,250), 2);
    cv::circle( curr_show, px_curr, 5, cv::Scalar(0,0,250), 2);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(1);
}

// show epipolar line 
void showEpipolarLine( const Mat& ref, const Mat& curr, const Point2f& px_ref, const Point2f& px_min_curr, const Point2f& px_max_curr ){
    Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );
    
    cv::circle( ref_show, px_ref, 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, px_min_curr, 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, px_max_curr, 5, cv::Scalar(0,255,0), 2);
    cv::line( curr_show, px_min_curr, px_max_curr, Scalar(0,255,0), 1);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(1);
}