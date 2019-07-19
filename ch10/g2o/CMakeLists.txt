cmake_minimum_required(VERSION 2.8)

project(global_ba)

set(CMAKE_BUILD_TYPE "Debug")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Cholmod REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall")

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

include_directories(${EIGEN3_INCLUDE_DIRS} ${CHOLMOD_INCLUDE_DIR} ${PEOJRCT_SOURCE_DIR}/ceres)
include_directories(
        ${PROJECT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/common/tools
        ${PEOJECT_SOURCE_DIR}/common/flags
        )

message("find g2o libraries in: " ${G2O_INCLUDE_DIR})
message("find g2o lib in " ${G2O_LIBS})
message("find cholmod in " ${CHOLMOD_INCLUDE_DIR})


add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

add_executable(g2o_bundle ./g2o_bundle.cpp)
target_link_libraries(g2o_bundle ${G2O_LIBS} ${CHOLMOD_LIBRARIES} BALProblem ParseCmd)
