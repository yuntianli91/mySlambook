cmake_minimum_required(VERSION 2.8)

project(direct_odometry)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
list(APPEND CMAKE_MODULE_PATH ../cmake_modules)

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    )

set(THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
    )

# add_executable(direct ./direct.cpp)
# target_link_libraries(direct ${THIRD_PARTY_LIBS})
add_executable(direct_sparse ./direct_sparse.cpp)
target_link_libraries(direct_sparse ${THIRD_PARTY_LIBS})
