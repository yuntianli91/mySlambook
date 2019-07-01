// head files from c++ std
#include <iostream>
#include <chrono>
// header file of g2o vertex
#include <g2o/core/base_vertex.h>
// header file of g2o edge
#include <g2o/core/base_unary_edge.h>
// header file of g2o sparse optimizer
// include GN, LM and Powell's doglog methods
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
 // header files for g2o solver
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
// header files of other packages
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

// Definition of your own vertex type
class CurveFittingVertex: public g2o::BaseVertex < 3, Eigen::Vector3d > {
  public:
    // Using this macro to align pointer in case of memory
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() {
      _estimate  = Eigen::Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double* update) {
      _estimate += Eigen::Vector3d(update);
    }

    // override read and write function
    // **required in all custom type of edge and vertex**
    virtual bool read(std::istream& in ) {}
    virtual bool write(std::ostream& out) const {}
};
// Definition of your own edge type
class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
  public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}
    // error function
      const CurveFittingVertex* v = static_cast <const CurveFittingVertex*> (_vertices[0]);
    virtual void computeError() {
      const Eigen::Vector3d abc = v->estimate();
      // error function f(x, y) = y - exp( ax^2 + bx + c)
      _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    virtual bool read( std::istream& in) {}
    virtual bool write( std::ostream& out) const {}
  public: double _x;
};

int main(int argc, char ** argv) {
  double a = 1.0, b = 2.0, c = 1.0;
  int N = 100;
  double w_sigma = 1.0;
  cv::RNG rng;
  double abc[3] = {0, 0, 0};

  std::vector<double> x_data, y_data;

  std::cout << "Generating simulation data...";

  for (int i=0; i< N; i++){
    double x = i / 100.;
    x_data.push_back(x);
    y_data.push_back(std::exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    std::cout << x_data[i] << " " << y_data[i] << std::endl;
  }
  
  
  // ----------------------------- Config g2o Solver ------------------------------------------------//
  // Definite the type of solver, parameters are <PoseDim, LandMarkDim>
  typedef g2o::BlockSolver < g2o::BlockSolverTraits < 3, 1 >> Block;
  // Step 1. Config linear solver (should use std::unique_ptr in new g2o version)
  // Create Linear Solver with specific SparseBlockMatrix type.
  Block::LinearSolverType* linearsolver = new g2o::LinearSolverDense < Block::PoseMatrixType > ();
  // Initiate Block Solver with linear solver.
  Block* block_ptr = new Block(std::unique_ptr < Block::LinearSolverType > (linearsolver));
  // Step 2. Config solver method: GN, LM or Dogleg.(Also notice the std::unique_ptr is using now.)
  // Uncomment the mehod you want to use.
  // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<Block>(block_ptr));
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr < Block > (block_ptr));
  // g2o::OptimizationAlgorithmDogleg* solver = new g2O::OptimizationAlgorithmDogleg(std::unique_ptr<Block>(block_ptr));
  // Step 3. Config sparse optimizable graph with defined solver
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);
  // ------------------------------------------------------------------------------------------------//
  
  // ------------------------------ Config optimizable graph ----------------------------------------//
  // Add vertics to hyper graph
  CurveFittingVertex* v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(0, 0, 0)); 
  v->setId(0);
  optimizer.addVertex(v);

  for (int i=0; i <N; i++){
    CurveFittingEdge* edge = new CurveFittingEdge( x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);
    edge->setMeasurement(y_data[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1/(w_sigma*w_sigma));
    optimizer.addEdge(edge);
  }
  // ------------------------------ Config g2o solver parameters ------------------------------------//
  std::cout << "start optimization" << std::endl;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "Solve time cost = " << time_cost.count() << std:: endl;

  Eigen::Vector3d abc_final = v->estimate();
  std::cout << "estimated model: " << abc_final.transpose() << std::endl; 
  
  return 0;

}