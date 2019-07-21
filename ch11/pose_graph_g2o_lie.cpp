/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-18 09:57:08
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-19 14:11:00
 */
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using namespace Sophus;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
// ================== custom functions ======================== //
Matrix6d Jr_inv(SE3d e_ij){
    Matrix6d J;

    // block(start_x, start_y, length_x, length_y)
    J.block(0, 0, 3, 3) = SO3d::hat(e_ij.so3().log()); 
    J.block(0, 3, 3, 3) = SO3d::hat(e_ij.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e_ij.so3().log());
    J = Matrix6d::Identity() + 0.5 * J;

    return J;

}
// ================== custom g2o types ======================== //
typedef Eigen::Matrix<double, 6, 1> Vector6d;
class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // 
    virtual bool read(istream& in){
        double data[7];
        for (int i = 0; i < 7; i ++) in >> data[i];
            setEstimate(SE3d(
                Quaterniond(data[6], data[3], data[4], data[5]),
                Vector3d(data[0], data[1], data[2])
            ));
    }

    virtual bool write(ostream& out) const
    {
        out << id() << " ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();

        out << _estimate.translation().transpose() << " ";
        out << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3];
        return true;
    }

    virtual void setToOriginImpl(){
        _estimate = SE3d();
    }

    virtual void oplusImpl(const double* update){
        Vector3d rvec(update[3], update[4], update[5]);
    
        SE3d delta(
            Quaterniond(AngleAxisd(rvec.norm(), rvec.normalized())),
            Vector3d(update[0], update[1], update[2]));

        _estimate = delta * _estimate;        
    }
};

class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual bool read(istream& in){
        double data[7];
        for (int i = 0; i < 7; i ++) in >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2])));

        for (int i = 0; i < information().rows() && in.good(); i ++){
            for (int j = i; j < information().cols() && in.good(); j++){
                in >> information()(i, j);
                if (i != j) information()(j, i) = information()(i, j);
            }
        }

        return true;
    }

    virtual bool write(ostream& out)const{
        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*>(_vertices[1]);

        out << v1->id() << " " << v2->id() << " ";
        SE3d m = _measurement;
        Quaterniond q = m.unit_quaternion();
        out << m.translation().transpose() << " ";
        out << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " <<q.coeffs()[3];

        for (int i = 0; i < information().rows(); i ++){
            for ( int j = i; j < information().cols(); j ++){
                out << information()(i, j) << " ";
            }
        }
        out << endl;
        return true;
    }

    virtual void computeError(){
        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*>(_vertices[1]);

        _error = (_measurement.inverse() * v1->estimate().inverse() * v2->estimate()).log();
    }

    virtual void linearizeOplus(){
        // VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*>(_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*>(_vertices[1]);
      
        Matrix6d J = Jr_inv(SE3d::exp(_error));

        _jacobianOplusXi = - J * v2->estimate().inverse().Adj();
        _jacobianOplusXj = J * v2->estimate().inverse().Adj();
    }

};



int main(int argc, char** argv){
 if (argc!=2){
        cout << "Usage:pose_graph_g2o_SE3 file.g2o." << endl;
        return 1;
    }
    // read file and check read status
    ifstream fin(argv[1]);
    if (!fin.is_open()){
        cout << "Failed to open g2o data file." << endl;
        return 1;
    }
    // config g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block; // pose and landmark are both 6d se3;
    Block::LinearSolverType* linear_solver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block* block_solver = new Block(unique_ptr<Block::LinearSolverType>(linear_solver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(block_solver));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    int vertexCount = 0, edgeCount = 0;

    vector<VertexSE3LieAlgebra*> pvertices;
    vector<EdgeSE3LieAlgebra*> pedges;
    
    while (fin.peek()!=EOF){
        string lineHead;
        fin >> lineHead;

        if(lineHead == "VERTEX_SE3:QUAT"){
            VertexSE3LieAlgebra* vertex = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;

            vertex->setId(index);
            vertex->read(fin);
            optimizer.addVertex(vertex);
            vertexCount++;
            if(index == 0)vertex->setFixed(true);
        }
        else if(lineHead == "EDGE_SE3:QUAT"){
            EdgeSE3LieAlgebra* edge = new EdgeSE3LieAlgebra();
            int id1, id2;
            fin >> id1 >> id2;
            edge->setId(edgeCount++);
            edge->setVertex(0, optimizer.vertices()[id1]);
            edge->setVertex(1, optimizer.vertices()[id2]);
            edge->read(fin);
            optimizer.addEdge(edge);           
        }
        if(!fin.good())break;
    }

    cout << "Read total " << vertexCount << " vertices and " << edgeCount << " edges." << endl;
    cout << "Initiating optimization ..." << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    cout << "Optimization started..." << endl;
    optimizer.optimize(30);
    cout << "Saving optimization results..." << endl;
    
    ofstream fout("result_lie.g2o");
    for ( VertexSE3LieAlgebra* v:pvertices )
    {
        fout<<"VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for ( EdgeSE3LieAlgebra* e:pedges )
    {
        fout<<"EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();

    return 0;
}
