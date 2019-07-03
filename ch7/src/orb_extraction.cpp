// standard c++ headers
#include <iostream>
#include <cmath>
#include <algorithm>
// third party headers
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    // check number of input parameters from cmd line
    if (argc!=3){
        cout << "Error ! No matching parameters list founded !" << endl
        << "Usage: orb_extraction file1.jpg file2.jpg";
        }

    // read img from files;
    Mat img1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    Mat img2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    // initiate feature detector
    // using smart pointer as opencv required  
    Ptr<ORB> orb = ORB::create();
    // Detect and show features
    vector<KeyPoint> keypoints1, keypoints2;
    orb->detect(img1, keypoints1);
    orb->detect(img2, keypoints2);

    Mat keypointImg;
    drawKeypoints(img1, keypoints1, keypointImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("Orb_feature", keypointImg);
    // generate descriptor for keypoints
    Mat descriptor1, descriptor2;
    orb->compute(img1, keypoints1, descriptor1);
    orb->compute(img2, keypoints2, descriptor2);
    //  Match descriptors 
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptor1, descriptor2, matches);
    Mat original_matches_img;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, original_matches_img );
    imshow("Original Orb Matches", original_matches_img);
    // Select matches
    double min_dist = 10000, max_dist = 0;

    for(int i = 0; i < descriptor1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist){min_dist = dist;}
        if(dist > max_dist){max_dist = dist;}
    }
    std::cout << "Minimum distance of descriptors is: " << min_dist << endl;
    std::cout << "Maximum distance of descriptors is: " << max_dist << endl;
 
    vector<DMatch> selected_matches;
    for(int i = 0; i < descriptor1.rows; i++){
        if (matches[i].distance < max(2 * min_dist, 30.) ){
            selected_matches.push_back(matches[i]);
        }
    }

    Mat selected_matches_img;
    drawMatches(img1, keypoints1, img2, keypoints2, selected_matches, selected_matches_img );
    imshow("Selected Orb Matches", selected_matches_img);
    waitKey(0);
    return 0;
}