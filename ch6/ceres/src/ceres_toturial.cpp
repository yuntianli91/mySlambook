#include <iostream>
#include <Eigen/Core>
#include <ceres/ceres.h>

// CostFunctor
struct CostFunctor{
  template <typename _T>
    bool operator()(const _T* const x, _T* residual) const{
      residual[0] = _T(10.0) - x[0];
      return true;
    }
};


int main(int argc, char** argv){
  // Initial value
  double init_x = 5.0 ;
  double x = init_x;
  // build problem
  ceres::Problem problem;
  ceres::CostFunction* cost_functon = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_functon, nullptr, &x);

  //config solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "initial x :" << init_x << std::endl
    << "optimized x :" << x << std::endl;
  return 0;

}
