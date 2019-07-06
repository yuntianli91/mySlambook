cmake_minimum_required(VERSION 2.8)

project(vo_feature)

list(APPEND CMAKE_MODULE_PATH ../cmake_module)

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3"
    )

set(THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
    )

add_executable(orb_extractor ./orb_extraction.cpp)
target_link_libraries(orb_extractor ${OpenCV_LIBS})

add_executable(pose_2d2d ./pose_2d2d.cpp)
target_link_libraries(pose_2d2d ${OpenCV_LIBS})

add_executable(triangulation ./triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_2d3d ./pose_2d3d.cpp)
target_link_libraries(pose_2d3d ${THIRD_PARTY_LIBS})

add_executable(pose_3d3d ./pose_3d3d.cpp)
target_link_libraries(pose_3d3d ${OpenCV_LIBS})

# add_executable(pose_estimation_3d2d ./pose_estimation_3d2d.cpp)
# target_link_libraries(pose_estimation_3d2d ${THIRD_PARTY_LIBS})
