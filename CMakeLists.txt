
cmake_minimum_required(VERSION 2.6)
project(VC LANGUAGES CUDA CXX)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${VC_SOURCE_DIR})

find_package(CUDA REQUIRED)
message(STATUS "CUDA_INCLUDE_DIR: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS	: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARY}")
message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARIES}")

#set(CMAKE_CXX_FLAGS "-O2 -g -Wall -Wextra -Wshadow -Werror -ansi")
set(CMAKE_CXX_FLAGS "-O3 -Wall -ffast-math -DNDEBUG")
#set(CUDA_NVCC_FLAGS -g -G)
set(CUDA_NVCC_FLAGS -Xptxas -dlcm=cg)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VC_SOURCE_DIR}/bin )
#add_subdirectory(${VC_SOURCE_DIR}/ppcsr/)
add_subdirectory(${VC_SOURCE_DIR}/src/)
#enable_testing()
#include(gtest/GoogleTest.cmake)
#add_subdirectory(gtest)


