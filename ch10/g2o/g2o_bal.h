/*
 * @Description: g2o ba types
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-16 09:33:43
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-17 11:18:20
 */
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>

#include "./ceres/autodiff.h"

#include "./common/tools/rotation.h"
#include "./common/projection.h"

class VertexCameraBAL : public g2o::BaseVertex<9, Eigen::VectorXd>
{
    // camera point vector     
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL(){}

    virtual bool read(std::istream& in){return false;}
    virtual bool write(std::ostream& out) const {return false;};

    virtual void setToOriginImpl(){
        _estimate = Eigen::VectorXd();
    }

    virtual void oplusImpl(const double* update){
        // convert C++ array to Eigen dynamic vector
        Eigen::VectorXd::ConstMapType v(update, VertexCameraBAL::Dimension);
        _estimate += v;
    }
};

class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
    // landmark point vector
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {} // constructor
    // virtual function
    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }
    // virtual tunction
    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( _vertices[0] );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( _vertices[1] );
        
        // *this return class object itself
        // this "equal" to EdgeObservationBAL(.....)
        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        // function from common/projection.h header
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );
        // ======================================= config ceres autodiff solver ======================================= //
        //AutoDiff <Functor, output_type, dimension of first variable, dimension of second variable, .....>
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;
        // J = [J_camera, J_Point] row_major
        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera; // J_camera
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint; // J_Point
        // use const_cast to remove const property
        double* parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double* jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};
