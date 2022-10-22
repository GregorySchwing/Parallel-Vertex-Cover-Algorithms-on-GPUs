cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(vc-gpu VERSION 0.1 LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -ansi -pedantic-errors -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

file(GLOB_RECURSE vc-gpu_SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE vc-gpu_HEADERS "${PROJECT_SOURCE_DIR}/*.h")

file(GLOB_RECURSE vc-gpu_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE vc-gpu_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/*.cuh")

set(vc-gpu_INCLUDE_DIRS "")
foreach (_headerFile ${vc-gpu_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list(APPEND vc-gpu_INCLUDE_DIRS ${_dir})
endforeach ()
list(REMOVE_DUPLICATES vc-gpu_INCLUDE_DIRS)

add_subdirectory(parallel-packed-csr)
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "vc-gpu_SOURCES: ${vc-gpu_SOURCES}")

add_executable(vc-gpu ${vc-gpu_CUDA_SOURCES} ${vc-gpu_SOURCES}  ${vc-gpu_HEADERS} ${vc-gpu_CUDA_HEADERS})
target_include_directories(vc-gpu PRIVATE ${vc-gpu_INCLUDE_DIRS})

