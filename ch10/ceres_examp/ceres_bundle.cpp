/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-17 14:02:56
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-18 09:10:54
 */
#include <iostream>
#include <fstream>

#include "reproject_error.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"

void setLinearSolver(ceres::Solver::Options* options, const BundleParams& params);
void setOrdering(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params);
void setMinimizerOptions(ceres::Solver::Options* options, const BundleParams& params);
void setSolverOptionsFromFlags(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params);
void buildProblem(BALProblem* bal_problem, ceres::Problem* problem, const BundleParams& params);
void solveProblem(const string filename, const BundleParams& params);

int main(int argc, char** argv){
    BundleParams params(argc,argv);  // set the parameters here.
   
    google::InitGoogleLogging(argv[0]);
    std::cout << params.input << std::endl;
    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    solveProblem(params.input.c_str(), params);
 
    return 0;
}

void setLinearSolver(ceres::Solver::Options* options, const BundleParams& params){
    // set linearsolvertype 
    CHECK(ceres::StringToLinearSolverType(params.linear_solver, &(options->linear_solver_type)));
    // ====!!! Notes: set sparse algebra and library or dense algebra library do not
    // ====!!! affect which kine of linear solver ceres use.

    // set sparse linear algebra library (like Eigen::Sparse) 
    // this library will be used when linear_solver_type has been set as sparse types like sparse_normal_cholesky
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library, &(options->sparse_linear_algebra_library_type)));

    // set dense linear algebre library (linke Eigen::Dense)
    // this library will be used when linear_solver_type has been set as dense types like Dense_QR
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library, &(options->dense_linear_algebra_library_type)));

    // this option has been removed since ceres 1.15.0, all threads are controled 
    // by Options::num_threads.
    // options->num_linear_solver_threads = params.num_threads; 
}
void setOrdering(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params){
    // set schur order
    const int num_cameras = bal_problem->num_points();
    const int num_points = bal_problem->num_points();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();
    
    double* cameras = bal_problem->mutable_cameras();
    double* points = bal_problem->mutable_points();

    if(params.ordering == "automatic") return;
    // create ParameterBlockOrdering object to config schur sequence
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering();

    // set all points in ordering to 0
    for(int i = 0; i < num_points; i++){
        ordering->AddElementToGroup(points + i * point_block_size, 0);
    }
    // set all cameras in ordering to 1
    for(int i = 0; i < num_cameras; i++){
        ordering->AddElementToGroup(cameras + i * camera_block_size, 1);
    }

    // set ordering in options
    options->linear_solver_ordering.reset(ordering);
 
    
}
void setMinimizerOptions(ceres::Solver::Options* options, const BundleParams& params){
    options->max_num_iterations = params.num_iterations; // set maximum iterator
    options->minimizer_progress_to_stdout = true; // output progress to std
    options->num_threads = params.num_threads; // multi threads to speed up optimization
    
    CHECK(ceres::StringToTrustRegionStrategyType(params.trust_region_strategy, &(options->trust_region_strategy_type)));

}
void setSolverOptionsFromFlags(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params){
    setLinearSolver(options, params);
    setMinimizerOptions(options, params);
    setOrdering(bal_problem, options, params);

}
void buildProblem(BALProblem* bal_problem, ceres::Problem* problem, const BundleParams& params){
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    double* points = bal_problem->mutable_points();
    double* cameras = bal_problem->mutable_cameras();

    const double* observations = bal_problem->observations();
    for(int i = 0; i < bal_problem->num_observations(); i ++){
        ceres::CostFunction* cost_function = ReprojectError::create(observations[2 * i + 0], observations[2 * i +1]); 
        // conditional expression: a ? b : c
        // equal to 'if a then b else c'
        ceres::LossFunction* loss_function = params.robustify ? new ceres::HuberLoss(1.0): NULL;
        // TODO:add camera and point
        double* pcamera = cameras + camera_block_size * bal_problem->camera_index()[i]; // return camera id corresponds to i observation 
        // cout << "camera_index of " << i << " is " << bal_problem->camera_index()[i] << endl;
        double* ppoint = points + point_block_size * bal_problem->point_index()[i];  // return point if corresponds to i observation
        
        problem->AddResidualBlock(cost_function, loss_function, pcamera, ppoint);

    }
}
void solveProblem(const string filename, const BundleParams& params){
    BALProblem bal_problem(filename);

         // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..
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
    
    ceres::Problem problem;
    buildProblem(&bal_problem, &problem, params);

    std::cout << "the problem is successfully build.." << std::endl;

    ceres::Solver::Options options;
    setSolverOptionsFromFlags(&bal_problem, &options, params);
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // write the result into a .ply file.   
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);  // pay attention to this: ceres doesn't copy the value into optimizer, but implement on raw data! 
    }
}