cmake_minimum_required(VERSION 2.8)

project("g2o fitting curve")
set( CMAKE_CXX_FLAGS "-std=c++14")
list( APPEND CMAKE_MODULE_PATH ./cmake_module)

find_package(OpenCV REQUIRED)

include_directories(
    "/usr/include/eigen3"
    ${G2O_INCLUDE_DIRS}
    ${OpenCV_DIRS}
)

add_executable(curve_fitting_g2o ./curve_fitting_g2o)
target_link_libraries(curve_fitting_g2o ${OpenCV_LIBS} g2o_core g2o_stuff)
