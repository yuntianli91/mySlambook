/*
 * @Description: training with fbow
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-20 11:09:00
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-21 10:46:07
 */
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>

using namespace std;

int main(int argc, char** argv){
    // check input argvs
    if (argc!=2) cout << "Usage: feature_training dataset_dir." << endl;
    // read dataset images;
    string dataset_dir = argv[1];
    ifstream fin(dataset_dir + "/rgb.txt");

    vector<double> time_stamps;
    vector<string> filenames;
    vector<cv::Mat> rgb_images;
    // read images from dataset
    cout << "loading images from dataset..." << endl;    
    while(fin.peek() != EOF){
        string line;
        getline(fin, line);
        if(line[0] == '#') continue;
        
        istringstream iss(line);
        double time_stamp;
        string filename;
        iss >> time_stamp >> filename;
        time_stamps.push_back(time_stamp);
        filenames.push_back(filename); 

        cv::Mat img = cv::imread(dataset_dir + "/" + filename, CV_LOAD_IMAGE_UNCHANGED); 
        rgb_images.push_back(img); 
    }
    cout << "Totally read " << rgb_images.size() << " images." << endl;
    // compute descriptors
    vector<cv::Mat> descriptors;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    
    cout << "detecting features and computing descriptors..." << endl;
    for (size_t i = 0; i < rgb_images.size(); i ++){
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        orb->detectAndCompute(rgb_images[i], cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }    

    // creating vocabulary
    fbow::VocabularyCreator::Params params;
    params.k = 10; // branch number
    params.L = 5; // layer number  
    // params.nthreads = 1; // threads
    // params.maxIters = 11; // max iterations
    // params.verbose = false; // display message
    fbow::VocabularyCreator voc_creator;
    fbow::Vocabulary voc;

    cout << "creating vocabulary with " << params.k << " branches and " << params.L << " layers..." <<endl;
    string desc_name = "ORB";
    
    voc_creator.create(voc, descriptors, desc_name, params);
    cout << "generated voc with " << voc.size() << endl;
    cout << "save voc to file..." << endl;
    voc.saveToFile("fbow_voc.fbow");
    return 0;
}