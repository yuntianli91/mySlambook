/*
 * @Description: 
 * @Author: Yuntian Li
 * @Github: https://github.com/yuntinali91
 * @Date: 2019-07-20 16:07:22
 * @LastEditors: Yuntian Li
 * @LastEditTime: 2019-07-21 09:23:42
 */
#include <iostream>
#include <fstream>
#include <map>
#include <fbow/fbow.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv){
 
    if (argc!=3) {
        cout << "not enough input arguments..." << endl;
        return 1;
    }
 
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

    fbow::Vocabulary voc;
    cout << "Reading vocabulary..." << endl;
    voc.readFromFile(argv[2]);
    string dest_name = voc.getDescName();
    cout << "dest_name: " << dest_name << endl;
    cout << "voc_size: " << voc.size() << endl;

    cout << "comparing images with images..." << endl;
    fbow::fBow v1, v2;
    for (size_t i = 0; i < descriptors.size(); i++){
        v1 = voc.transform(descriptors[i]);
        for (size_t j = 0; j < descriptors.size(); j++){
            v2 = voc.transform(descriptors[j]);
            double score = v1.score(v1, v2);
            cout << "image " << i << " vs image " << j << " : " << score <<endl;
        }

    }

    cout << "comparing images with database..." << endl;
    
}