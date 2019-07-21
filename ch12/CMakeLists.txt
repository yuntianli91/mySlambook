cmake_minimum_required(VERSION 2.8)

project(loop_closure)

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -Wall -O3")
list( APPEND CMAKE_MODULE cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(fbow REQUIRED)
include_directories(${fbow_INCLUDE_DIRS})

add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${fbow_LIBS})

add_executable(feature_compare feature_compare.cpp)
target_link_libraries(feature_compare ${OpenCV_LIBS} ${fbow_LIBS})