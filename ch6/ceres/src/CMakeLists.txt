cmake_minimum_required( VERSION 2.8 )
project( ceres_tutorial )
#-------------- config cmake --------------#
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++14" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
#-------------- config cers ---------------#
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )
#-------------- config opencv--------------#
set( OpenCV_DIR "/usr/local/opencv3.4/share/OpenCV" )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#-------------- config pcl-----------------#
find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( ceres_tutorial ./ceres_toturial.cpp )
target_link_libraries( ceres_tutorial ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( curve_fitting ./curve_fitting.cpp )
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )
