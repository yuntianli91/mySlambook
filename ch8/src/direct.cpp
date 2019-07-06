// c++ headers
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
// third party headers
#include <opencv2/opencv.hpp>
// g2o headers
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

using namespace std;
// ---------------------------------------- Custom function declarations ------------------------------- //
// keypoint mesaurement in keyframe
struct Measurement{
    Measurement(Eigen::Vector3d pt_w, float grayscale):pt_w_(pt_w), grayscale_(grayscale){}
    Eigen::Vector3d pt_w_;
    float grayscale_;
};
// ---------------------------------------- Edge of photometric error ---------------------------------- //
class EdgeSE3Photometric: public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>{
    // ============================== functions ===================================== //
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // constructor
        EdgeSE3Photometric(){}
        EdgeSE3Photometric(Eigen::Vector3d& point, const Eigen::Matrix3d& K, const cv::Mat& image):point_w(point), K_(K), image_(image){}
        // error computation function
        virtual void computeError(){
            const g2o::VertexSE3Expmap* vi = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
            // project point_world to point_camera
            Eigen::Vector3d point_c = vi->estimate().map(point_w);
            // convert from point_camera to point_pixel
            Eigen::Vector3d point_p_uniform = (1 / point_c(2,0)) * K_ * point_c;
            Eigen::Vector2d point_p(point_p_uniform.head(2));
            // check whether point_p is inside the image;
            float u = point_p(0, 0);
            float v = point_p(1, 0);
            if(u - 4 < 0 || u + 4 > image_.cols 
                || v - 4 < 0 || v + 4 > image_.rows) {
                _error(0, 0) = 0.0;
                // level 1 means outlier, level 0 means inlier
                // only inlier will be optimized 
                this->setLevel(1);
            }
            else{
                _error(0, 0) = getSubPixValue(u, v) - _measurement;
            }
        }
        // gradiant computation function
        virtual void linearizeOplus(){
            
            if (level() == 1){
                _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
                return;
            }

            const g2o::VertexSE3Expmap* vi = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
            // project point_world to point_camera
            Eigen::Vector3d point_c = vi->estimate().map(point_w);
            double x = point_c(0, 0);
            double y = point_c(1, 0);
            double z = point_c(2, 0);
            // convert from point_camera to point_pixel
            Eigen::Vector3d point_p_uniform = (1 / point_c(2,0)) * K_ * point_c;
            Eigen::Vector2d point_p(point_p_uniform.head(2));
            float u = point_p(0, 0);
            float v = point_p(1, 0);

            // partial I with respect to partial u
            // e.g. image gradiant
            Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
            jacobian_pixel_uv(0, 0) = (getSubPixValue(u + 1, v) - getSubPixValue(u - 1, v)) / 2;
            jacobian_pixel_uv(0, 1) = (getSubPixValue(u, v + 1) - getSubPixValue(u, v - 1)) / 2;
            // partial uv with partial ksai
            Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
            jacobian_uv_ksai(0, 0) =  - x * y / (z * z) * K_(0, 0); 
            jacobian_uv_ksai(0, 1) = (1+ (x * x / (z * z))) * K_(0, 0); 
            jacobian_uv_ksai(0, 2) =  - y / z * K_(0, 0); 
            jacobian_uv_ksai(0, 3) = 1 / z * K_(0, 0); 
            jacobian_uv_ksai(0, 4) = 0; 
            jacobian_uv_ksai(0, 5) =  - x / (z * z) * K_(0, 0); 
        
            jacobian_uv_ksai(1, 0) = -(1 + y * y / (z * z)) * K_(1, 1); 
            jacobian_uv_ksai(1, 1) = x * y / (z * z) * K_(1, 1); 
            jacobian_uv_ksai(1, 2) = x / z * K_(1,1); 
            jacobian_uv_ksai(1, 3) = 0; 
            jacobian_uv_ksai(1, 4) = 1 / z * K_(1, 1); 
            jacobian_uv_ksai(1, 5) = - y / (z * z) * K_(1, 1); 
            //
            _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai; 

        }
        // dummy read and write function
        virtual bool read(std::istream& in) {}
        virtual bool write(std::ostream& out) const {}
    protected:
        float getSubPixValue(float u, float v){
            // use bilinear interpolation to obtain subpix value
            int x = floor(v);
            int y = floor(u);
            float dx = v - floor(v);
            float dy = u - floor(u);

            return float( (1 - dx) * (1 - dy) * image_.at<uchar>(x, y)
                        + dx * (1 - dy) * image_.at<uchar>(x, y + 1) 
                        + (1 - dx) * dy * image_.at<uchar>(x + 1, y)
                        + dx * dy * image_.at<uchar>(x + 1, y + 1));
        }
    //  ============================= variables ===================================== //
    public:
        Eigen::Vector3d point_w;
        Eigen::Matrix3d K_;
        cv::Mat image_;
};
// Eigen::Vector3d project2Dto3D ( float x, float y, int d, float fx, float fy, float cx, float cy, float scale )
// {
//     float zz = float ( d ) /scale;
//     float xx = zz* ( x-cx ) /fx;
//     float yy = zz* ( y-cy ) /fy;
//     return Eigen::Vector3d ( xx, yy, zz );
// }
// ---------------------------------------- Custom function definitions -------------------------------- //
void project2d3d(const Eigen::Vector2d& pt_uv, int d, Eigen::Vector3d& pt_w, const Eigen::Matrix3d& K, const float depth_scale){
// project uv to p
    pt_w(2, 0) = (double)d / (double)depth_scale;
    // x = (u-cx) / fx * z;
    pt_w(0, 0) = pt_w(2, 0) * (pt_uv(0, 0) - K(0, 2)) / K(0, 0);
    // y = (v-cy) / fy * z
    pt_w(1, 0) = pt_w(2, 0) * (pt_uv(1, 0) - K(1, 2)) / K(1, 1);
}


void project3d2d(const Eigen::Vector3d& pt_w, Eigen::Vector2d& pt_uv, const Eigen::Matrix3d& K);
void poseEstimatorDirect(const vector<Measurement>& kp_reference, const cv::Mat& img, const Eigen::Matrix3d& K,const Eigen::Isometry3d& Tcw);
void directDisplay(const cv::Mat& kyFrame, const cv::Mat& cFrame, const vector<Measurement>& kp_reference, const Eigen::Isometry3d& Tcw, const Eigen::Matrix3d& K);
// ---------------------------------------- Main function ---------------------------------------------- //
int main(int argc, char** argv){
    
    if(argc != 2){
        cout << "Error: not enought input arguments." << endl
                <<"Usage: direct <dataset_path>." << endl;
        return 1;
    }

    string dataset_path = argv[1];
    string associate_path = dataset_path + "/associate.txt";

    ifstream fin(associate_path);   
    string rgb_file, depth_file, rgb_time, depth_time;

    // config parameters
    // camera parameters
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    // set transformation matrix of first frame as I
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();  

    cv::Mat img, depth;
    // reference keypoints in keyframe
    vector<Measurement> kp_reference;
    cv::Mat kyFrame;
    for (int i = 0; i < 10; i++){
        // loading rgb and depth image from files
        fin >> depth_time >> depth_file >> rgb_time >> rgb_file;
        img = cv::imread(rgb_file, CV_LOAD_IMAGE_GRAYSCALE);
        depth = cv::imread(depth_file, CV_LOAD_IMAGE_UNCHANGED);
        // check image read is fine
        if (img.data == nullptr || depth.data == nullptr){
            continue;
        }
        // using first frame as the keyframe and calculate information
        // ------------------------------------------------------------------------- //
        vector<cv::KeyPoint> keypoints_kf;
        if (i == 1){
            kyFrame = img.clone();
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();  
            detector->detect(img, keypoints_kf);
            vector<Eigen::Vector3d> measurements;
            for(cv::KeyPoint kp:keypoints_kf){
                // remove keypoints too close to image bouder
                if ( kp.pt.x < 20 || kp.pt.y < 20 
                    || ( kp.pt.x + 20 ) > img.cols || ( kp.pt.y + 20 ) < img.rows){continue;}
                // remove ketpoints with 0 depth
                int d = depth.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                if ( d == 0 ){continue;}
                Eigen::Vector2d pt_uv(kp.pt.x, kp.pt.y);
                Eigen::Vector3d pt_w;
                project2d3d(pt_uv, d, pt_w, K, depth_scale);

                float grayscale = (float)img.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
                kp_reference.push_back(Measurement(pt_w, grayscale));
            }
            continue;
        }
        // --------------------------------------------------------------------------- //
        // calculate pose with respect to keyframe
        // poseEstimatorDirect(kp_reference, img, K, Tcw);
        // directDisplay(kyFrame, img, kp_reference, Tcw, K);
    }
    return 0;
}

void project3d2d(const Eigen::Vector3d& pt_w, Eigen::Vector2d& pt_uv, const Eigen::Matrix3d& K){
    Eigen::Vector3d pt_uv_uniform;
    pt_uv_uniform = (1 / pt_w(2, 0)) * K * pt_w;
    pt_uv(0, 0) = pt_uv_uniform(0, 0);
    pt_uv(1, 0) = pt_uv_uniform(1, 0);
}

void poseEstimatorDirect(const vector<Measurement>& kp_reference, const cv::Mat& img, const Eigen::Matrix3d& K,const Eigen::Isometry3d& Tcw){
    // initiate sparse optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> Block;
    Block::LinearSolverType* solver_type = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(solver_type));
    g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // add vertex 
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    optimizer.addVertex(pose);
    // add edges
    int index = 1;
    for (Measurement kp:kp_reference){
        EdgeSE3Photometric* edge = new EdgeSE3Photometric(kp.pt_w_, K, img);
        edge->setId(index);
        edge->setVertex(0, pose);
        edge->setMeasurement(kp.grayscale_);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

        optimizer.addEdge(edge);
        index++;
    }

    cout << "total " << optimizer.edges().size() << " edges in optimizable graph." << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(30);
}

void directDisplay(const cv::Mat& kyFrame, const cv::Mat& cFrame, const vector<Measurement>& kp_reference, const Eigen::Isometry3d& Tcw, const Eigen::Matrix3d& K){
    // display features by direct estimated post
    cv::Mat imgToShow = cv::Mat(kyFrame.rows * 2, kyFrame.cols, CV_8UC3);
    kyFrame.copyTo(imgToShow(cv::Rect(0, 0, kyFrame.cols, kyFrame.rows)));
    cFrame.copyTo(imgToShow(cv::Rect(0, kyFrame.rows, cFrame.cols, cFrame.rows)));

    Eigen::Vector2d pt_kf, pt_c;

    for(Measurement kp:kp_reference){
        project3d2d(kp.pt_w_, pt_kf, K); 
        project3d2d(Tcw * kp.pt_w_, pt_c, K);
        // remove points out of bounders
        if (pt_c(0, 0) < 0 || pt_c(0, 0) > cFrame.cols 
            || pt_c(1, 0) < 0 || pt_c(1, 0) > cFrame.rows){continue;}  
        cv::circle(imgToShow, cv::Point2d(pt_kf(0, 0), pt_kf(1,0)), 8, cv::Scalar(-1), 2);
        cv::circle(imgToShow, cv::Point2d(pt_c(0, 0), kyFrame.rows + pt_c(1, 0)), 8, cv::Scalar(-1), 2);
        cv::line(imgToShow, cv::Point2d(pt_kf(0, 0), pt_kf(1,0)), cv::Point2d(pt_c(0, 0), kyFrame.rows + pt_c(1, 0)), cv::Scalar(-1), 1);
    }

    cv::imshow("direct results", imgToShow);
    cv::waitKey(0);
}