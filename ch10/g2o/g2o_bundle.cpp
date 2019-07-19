/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-16 11:00:32
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-17 12:40:18
 */
// c++ headers 
#include <iostream>

// g2o headers
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/core/robust_kernel_impl.h>
// custom headers
#include "common/BALProblem.h" // class for opertaions about file read nad write
#include "common/BundleParams.h" // class for operations about optimization parameters
#include "g2o_bal.h"

using namespace Eigen;
using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef; // VectorRef
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef; // readonly VectorRef
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3>> BalBlockSolver; // g2o block solver with pose(9d) and lanmark(3d)

// ================================ function for building Bal problem ====================================
// 
void BuildProblem(const BALProblem* bal_problem, g2o::SparseOptimizer* optimizer, const BundleParams& params){
    const int num_points = bal_problem->num_points(); // number of landmark points
    const int num_cameras = bal_problem->num_cameras(); // number of camera pose
    const int camera_block_size = bal_problem->camera_block_size(); // camera pose size in memory
    const int point_block_size = bal_problem->point_block_size(); // landmark point size in memory
    const int num_observations = bal_problem->num_observations(); //number of observations

    // add camera pose vertexs to hyper graph
    const double* cameras = bal_problem->cameras(); // return pointer of first camera pose

    for (int i = 0; i < num_cameras; i++){
        // save camera pose from c++ array to Eigen Vector using pre-defiend map
        // format: mapname(point, size)
        ConstVectorRef cameraVec(cameras + i * camera_block_size, camera_block_size);

        // construct camera pose vertex
        VertexCameraBAL* pose = new VertexCameraBAL();
        pose->setId(i);
        pose->setEstimate(cameraVec);
        // add vertex        
        optimizer->addVertex(pose);        
    }

    // add landmark point vertexs to hyper graph
    const double* points = bal_problem->points(); // return pointer of first landmark point

    for (int i = 0; i < num_points; i++){
        //save landmark point from c++ array to Eigen Vector 
        ConstVectorRef pointVec(points + i * point_block_size, point_block_size);

        // construct point vertex
        VertexPointBAL* point = new VertexPointBAL();
        point->setId(num_cameras + i);
        point->setEstimate(pointVec);

        point->setMarginalized(true);
        optimizer->addVertex(point);
    }

    // add edges
    const double* observations = bal_problem->observations();

    for (int i = 0; i < num_observations; i ++){
        EdgeObservationBAL* edge = new EdgeObservationBAL();
        
        edge->setId(i);

        const int camera_id = bal_problem->camera_index()[i]; // obtain id of camera pose vertex 
        const int point_id = bal_problem->point_index()[i] + num_cameras; // optain id of point vertex

        // use robust kernel funcion if edge is not stable
        if(params.robustify){
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            rk->setDelta(1.0);
            edge->setRobustKernel(rk);
        }
        // set vertexs for the edge
        edge->setVertex(0, dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
        edge->setVertex(1, dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
        edge->setInformation(Eigen::Matrix2d::Identity());
        // TODO set measurement 
        edge->setMeasurement(Eigen::Vector2d(observations[2 * i +0], observations[2 * i + 1]));

        optimizer->addEdge(edge);

    }     





}

void WriteToBALProblem(BALProblem* bal_problem, g2o::SparseOptimizer* optimizer){
    // write optimization bresults to BALProblem object
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    // obtain pointer of first camera
    // mutable_camera() return mutable pointer while camera() return const pointer
    double* cameras = bal_problem->mutable_cameras();
    for (int i = 0; i < num_cameras; i++){
        VertexCameraBAL* cameraVec = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd poseNew = cameraVec->estimate();
        // save data from optimizer to BALProblem
        // memcpy(dst_address, src_address, length of byte)
        memcpy(cameras + camera_block_size * i, poseNew.data(), sizeof(double) * camera_block_size);
    }

    double* points = bal_problem->mutable_points();
    for (int i=0; i < num_points; i++){
        VertexPointBAL* pointVec = dynamic_cast<VertexPointBAL*>(optimizer->vertex(i + num_cameras));
        Eigen::Vector3d pointNew = pointVec->estimate();
        memcpy(points + i * point_block_size, pointNew.data(), sizeof(double) * point_block_size);
    }
}

void SetMinimizerOptions(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params, g2o::SparseOptimizer* optimizer){
    // function to set solver parameters
    g2o::OptimizationAlgorithmWithHessian* solver;
    if (params.trust_region_strategy == "levenberg_marquardt"){
        solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BalBlockSolver>(solver_ptr.get()));
    }
    else if(params.trust_region_strategy == "dogleg"){
        solver = new g2o::OptimizationAlgorithmDogleg(unique_ptr<BalBlockSolver>(solver_ptr.get()));
    }
    else{
        cout << "Please check your trust_region_strategy parameter again.." << endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);
}

void setLinearSolver(std::shared_ptr<BalBlockSolver>& solver_ptr, const BundleParams& params){
    
    BalBlockSolver::LinearSolverType* linearSolver = nullptr;

    if(params.linear_solver == "dense_schur"){
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    }
    else if(params.linear_solver == "sparse_schur"){
        linearSolver = new g2o::LinearSolverCSparse<BalBlockSolver::PoseMatrixType>();
        dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>*>(linearSolver)->setBlockOrdering(true);
    }
    else{
        cout << "Please check your linear solver parameter again.." << endl;
        exit(EXIT_FAILURE);
    }

    solver_ptr = std::make_shared<BalBlockSolver>(unique_ptr<BalBlockSolver::LinearSolverType>(linearSolver));   
}

void solveProblem(const string filename, const BundleParams& params){
    BALProblem bal_problem(filename);

     // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    //  store initial 3D cloud points
    if(!params.initial_ply.empty()){
        bal_problem.WriteToPLYFile(params.initial_ply);
    }

     std::cout << "beginning problem..." << std::endl;
    
    // add some noise for the intial value
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;

    g2o::SparseOptimizer optimizer;
    shared_ptr<BalBlockSolver> blocksolver = nullptr;
    setLinearSolver(blocksolver, params);
    SetMinimizerOptions(blocksolver, params, &optimizer);
    BuildProblem(&bal_problem, &optimizer, params);

    std::cout << "begin optimizaiton .."<< std::endl;
    // perform the optimizaiton 
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);

    std::cout << "optimization complete.. "<< std::endl;
    // write the optimized data into BALProblem class
    WriteToBALProblem(&bal_problem, &optimizer);

    // write the result into a .ply file.
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);
    }
    
}

int main(int argc, char** argv)
{
  
    BundleParams params(argc,argv);  // set the parameters here.

    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    solveProblem(params.input.c_str(), params);

    return 0;
}