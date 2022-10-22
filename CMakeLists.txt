cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(vc-gpu VERSION 0.1 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -ansi -pedantic-errors -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE vc-gpu_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE vc-gpu_HEADERS "${PROJECT_SOURCE_DIR}/src/*.h")

file(GLOB_RECURSE vc-gpu_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE vc-gpu_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/src/*.cuh")

set(vc-gpu_INCLUDE_DIRS "")
foreach (_headerFile ${vc-gpu_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list(APPEND vc-gpu_INCLUDE_DIRS ${_dir})
endforeach ()
list(REMOVE_DUPLICATES vc-gpu_INCLUDE_DIRS)

add_subdirectory(parallel-packed-csr)

add_executable(vc-gpu ${vc-gpu_SOURCES} ${vc-gpu_CUDA_SOURCES})
target_include_directories(vc-gpu PRIVATE ${vc-gpu_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(vc-gpu PRIVATE Threads::Threads numa)

