/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-18 09:56:35
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-19 10:34:50
 */

#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
using namespace std;

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

    while (fin.peek()!=EOF){
        string lineHead;
        fin >> lineHead;

        if(lineHead == "VERTEX_SE3:QUAT"){
            g2o::VertexSE3* vertex = new g2o::VertexSE3();
            int index = 0;
            fin >> index;

            vertex->setId(index);
            vertex->read(fin);
            optimizer.addVertex(vertex);
            vertexCount++;
            if(index == 0)vertex->setFixed(true);
        }
        else if(lineHead == "EDGE_SE3:QUAT"){
            g2o::EdgeSE3* edge = new g2o::EdgeSE3();
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
    optimizer.save("results_SE3.g2o");

    return 0;
}