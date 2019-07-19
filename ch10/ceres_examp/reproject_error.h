/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-17 13:48:05
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-17 16:47:01
 */
#ifndef ReprojectError_H_
#define ReprojectError_H_

#include <iostream>
#include <ceres/ceres.h> 

#include "common/tools/rotation.h"
#include "common/projection.h"

class ReprojectError
{
    public:
    // consructor
    ReprojectError(double observation_x, double observation_y):observed_x_(observation_x), observed_y_(observation_y){}
    // override operator()
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals)const{
        // camera pose store as rotation vector
        T predictions[2];
        // reproject world point to uv
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x_);
        residuals[1] = predictions[1] - T(observed_y_);

        return true;
    }

    static ceres::CostFunction* create(const double observation_x, const double observation_y){
        return (new ceres::AutoDiffCostFunction<ReprojectError, 2, 9, 3>(
                new ReprojectError(observation_x, observation_y)));    
    }

    private:
    double observed_x_;
    double observed_y_;
};


#endif //ReprojectError.h
