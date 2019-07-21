/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-18 09:57:06
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-19 16:47:12
 */

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

using namespace std;
using namespace Sophus;

int main(int argc, char** argv){
    if (argc!=2){
        cout << "Usage: pose_graph_gtsam sephere.g2o." << endl;
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin){
        cout << "Failed to open " << argv[1] << ", maybe file path is wrong." << endl;
        return 1;
    }

    gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph); // shared pointer to factor graph object
    gtsam::Values::shared_ptr initial(new gtsam::Values); // shared pointer to initial values

    int countVar = 0, countFactor = 0; // factor is equal to edge, valurable is equal to vertex 

    while(fin.peek()!= EOF){
        string lineHead;
        fin >> lineHead;
        if (lineHead == "VERTEX_SE3:QUAT"){

            gtsam::Key id; // gtsam type for ID
            fin >> id;

            double data[7];
            for (int i = 0; i < 7; i ++) fin >> data[i];
            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            initial->insert(id, gtsam::Pose3(R, t));

            countVar++;            
        }
        else if(lineHead == "EDGE_SE3:QUAT"){
            gtsam::Key id1, id2;
            fin >> id1 >> id2;
            
            double data[7];
            for (int i = 0; i < 7; i ++) fin >> data[i];
            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);

            gtsam::Matrix m = gtsam::I_6x6;
            for (int i = 0 ; i < 6; i ++){
                for (int j = i; j < 6; j ++){
                    double mij;
                    fin >> mij;
                    m(i, j) = mij;
                    m(j, i) = mij;
                }
            }

            gtsam::Matrix mgtsam = gtsam::I_6x6;
            // g2o diag elements of information matrix are first translation then rotation
            // gtsam diag elements of information matrix are first rotation then trsnalation
            mgtsam.block<3,3>(0, 0) = m.block<3,3>(3, 3); // cov rotation
            mgtsam.block<3,3>(3, 3) = m.block<3,3>(0, 0); // cov translation
            mgtsam.block<3,3>(0, 3) = m.block<3,3>(0, 3); // off diag
            mgtsam.block<3,3>(3, 0) = m.block<3,3>(3, 0); // off diag
 
            gtsam::SharedNoiseModel model = gtsam::noiseModel::Gaussian::Information(mgtsam);

            gtsam::NonlinearFactor::shared_ptr factor(
                new gtsam::BetweenFactor<gtsam::Pose3>(id1, id2, gtsam::Pose3(R, t), model)
            );

            graph->push_back(factor); // add factor to graph
            countFactor++;
        }

        if (!fin.good()) break;
    }

    cout << "read total: " << countVar << " variable vertices and " << countFactor << " factors." << endl;

    // add prior factor
    gtsam::Vector6 var; // variances of prior
    var << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(var);
    
    gtsam::Key firstID = 0;
 
    cout << "adding prior to graph..." << endl;
    for (const gtsam::Values::ConstKeyValuePair& keyValuePair:*initial){
        graph->add(gtsam::PriorFactor<gtsam::Pose3>(
            keyValuePair.key, keyValuePair.value.cast<gtsam::Pose3>(), priorModel)); // use cast<> due to value are gtsam::value type not gtsam::Pose3 type
            break;
    }

    cout << "optimizing the factor graph..." << endl;
    // config optimizer
    gtsam::LevenbergMarquardtParams params_lm;
    params_lm.setVerbosity("ERROR"); // which kind of information will be print during optimization
    params_lm.setMaxIterations(20);
    params_lm.setLinearSolverType("MULTIFRONTAL_QR");
    
    // optimization
    gtsam::LevenbergMarquardtOptimizer optimizer_LM(*graph, *initial, params_lm);

    gtsam::Values result = optimizer_LM.optimize();
    cout<<"Optimization complete"<<endl;
    cout<<"initial error: "<<graph->error ( *initial ) <<endl;
    cout<<"final error: "<<graph->error ( result ) <<endl;

    cout<<"done. write to g2o ... "<<endl;

     // 写入 g2o 文件，同样伪装成 g2o 中的顶点和边，以便用 g2o_viewer 查看。
    // 顶点咯
    ofstream fout ( "result_gtsam.g2o" );
    for ( const gtsam::Values::ConstKeyValuePair& key_value: result )
    {
        gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
        gtsam::Point3 p = pose.translation();
        gtsam::Quaternion q = pose.rotation().toQuaternion();
        fout<<"VERTEX_SE3:QUAT "<<key_value.key<<" "
            <<p.x() <<" "<<p.y() <<" "<<p.z() <<" "
            <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" "<<endl;
    }
    // 边咯 
    for ( gtsam::NonlinearFactor::shared_ptr factor: *graph )
    {
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr f = dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>( factor );
        if ( f )
        {
            gtsam::SharedNoiseModel model = f->noiseModel();
            gtsam::noiseModel::Gaussian::shared_ptr gaussianModel = dynamic_pointer_cast<gtsam::noiseModel::Gaussian>( model );
            if ( gaussianModel )
            {
                // write the edge information 
                gtsam::Matrix info = gaussianModel->R().transpose() * gaussianModel->R();
                gtsam::Pose3 pose = f->measured();
                gtsam::Point3 p = pose.translation();
                gtsam::Quaternion q = pose.rotation().toQuaternion();
                fout<<"EDGE_SE3:QUAT "<<f->key1()<<" "<<f->key2()<<" "
                    <<p.x() <<" "<<p.y() <<" "<<p.z() <<" "
                    <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" ";
                gtsam::Matrix infoG2o = gtsam::I_6x6;
                infoG2o.block(0,0,3,3) = info.block(3,3,3,3); // cov translation
                infoG2o.block(3,3,3,3) = info.block(0,0,3,3); // cov rotation
                infoG2o.block(0,3,3,3) = info.block(0,3,3,3); // off diagonal
                infoG2o.block(3,0,3,3) = info.block(3,0,3,3); // off diagonal
                for ( int i=0; i<6; i++ )
                    for ( int j=i; j<6; j++ )
                    {
                        fout<<infoG2o(i,j)<<" ";
                    }
                fout<<endl;
            }
        }
    }
    fout.close();
    cout<<"done."<<endl;
    return 0;
}
