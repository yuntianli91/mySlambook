cmake_minimum_required(VERSION 2.8)

project(vo_feature)

find_package(OpenCV REQUIRED)
set( CMAKE_BUILD_TYPE "Debug" )

include_directories(OpenCV_DIRS)

add_executable(orb_extractor ./orb_extraction.cpp)
target_link_libraries(orb_extractor ${OpenCV_LIBS})

add_executable(pose_2d2d ./pose_2d2d.cpp)
target_link_libraries(pose_2d2d ${OpenCV_LIBS})

add_executable(triangulation ./triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_2d3d ./pose_2d3d.cpp)
target_link_libraries(pose_2d3d ${OpenCV_LIBS})

add_executable(pose_3d3d ./pose_3d3d.cpp)
target_link_libraries(pose_3d3d ${OpenCV_LIBS})
