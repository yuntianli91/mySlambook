#include <iostream>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
using namespace std;
// 数据模拟函数
void data_generator(vector<double>& param, int num, double step, double sigma, vector<double>& x_data, vector<double>& y_data);

// 声明单价函数factor
struct fittingResidual{
  fittingResidual(double x, double y):x_(x),y_(y){}
  template <typename _T>
    bool operator()(const _T* const param, _T* residual) const{
      residual[0] = _T(y_) - exp(param[0] * _T(x_) + param[1]);
      return true;
    }
  private:
    const double x_;
    const double y_;
};

int main(int argc, char** argv){
  // 生成带噪声的随机数据点集
  double step = 0.1; int num = 200;
  vector<double> param = {0.3, 0.1};
  double sigma = 0.2;

  vector<double> x_data, y_data;
  data_generator(param, num, step, sigma, x_data, y_data);
  // 创建problem
  ceres::Problem problem;
  // 创建costfunction对象, 由于最终目标函数是所有数据点的误差平方和
  // 因此需要添加N个残差模块
  double param_est[2] = {0, 0};
  for(int i=0; i<num ; i++){
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<fittingResidual, 1, 2>(new fittingResidual(x_data[i], y_data[i]));
    problem.AddResidualBlock(cost_function, nullptr, param_est);
  }
  // 配置求解器
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;
  cout << "Read param are " << param[0] << " and " << param[1] << endl;
  cout << "Fitting param are " << param_est[0] << " and " << param[1] << endl;
}

void data_generator(vector<double>& param, int num, double step, double sigma, vector<double>& x_data, vector<double>& y_data){
  cv::RNG rng;
  
  cout << "Generating data..." << endl;
  for(int i=0; i<num; i++){
    double x = double(i * step);
    x_data.push_back(double(x));
    y_data.push_back(exp(param[0] * x + param[1]) + rng.gaussian(sigma));
  }
  cout << "Data generated !" << endl;
}
