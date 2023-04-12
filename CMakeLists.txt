cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(vc-gpu VERSION 0.1 LANGUAGES CXX CUDA)
# CUDA runtime

find_package(CUDAToolkit REQUIRED)
find_package(Boost 1.40 REQUIRED)

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
file(GLOB_RECURSE maxmatch_SOURCES "${PROJECT_SOURCE_DIR}/matching/*.cpp")
file(GLOB_RECURSE mis_SOURCES "${PROJECT_SOURCE_DIR}/mis/*.cpp")
file(GLOB_RECURSE vc-gpu_HEADERS "${PROJECT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE edmonds_SOURCES "${PROJECT_SOURCE_DIR}/edmonds/*.c")
file(GLOB_RECURSE edmonds_HEADERS "${PROJECT_SOURCE_DIR}/edmonds/*.h")

file(GLOB_RECURSE vc-gpu_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE maxmatch_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/matching/*.cu")
file(GLOB_RECURSE mis_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/mis/*.cu")
file(GLOB_RECURSE vc-gpu_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/*.cuh")
file(GLOB_RECURSE maxmatch_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/matching/*.cu")
file(GLOB_RECURSE mis_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/mis/*.h")
file(GLOB_RECURSE edmonds_CUDA_SOURCES "${PROJECT_SOURCE_DIR}/edmonds/*.cu")
file(GLOB_RECURSE edmonds_CUDA_HEADERS "${PROJECT_SOURCE_DIR}/edmonds/*.cuh")


set(vc-gpu_INCLUDE_DIRS "")
foreach (_headerFile ${vc-gpu_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list(APPEND vc-gpu_INCLUDE_DIRS ${_dir})
endforeach ()
list(REMOVE_DUPLICATES vc-gpu_INCLUDE_DIRS)

#add_subdirectory(parallel-packed-csr)
#file(GLOB_RECURSE pppcsr_HEADERS "${CMAKE_SOURCE_DIR}/parallel-packed-csr/src/*.h")

#set(pppcsr_INCLUDE_DIRS "")
#foreach (_headerFile ${pppcsr_HEADERS})
#    get_filename_component(_dir ${_headerFile} PATH)
#    list(APPEND pppcsr_INCLUDE_DIRS ${_dir})
#endforeach ()
#list(REMOVE_DUPLICATES pppcsr_INCLUDE_DIRS)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "vc-gpu_SOURCES: ${vc-gpu_SOURCES}")
message(STATUS "maxmatch_SOURCES: ${maxmatch_SOURCES}")
message(STATUS "mis_SOURCES: ${mis_SOURCES}")

message(STATUS "pppcsr_INCLUDE_DIRS: ${pppcsr_INCLUDE_DIRS}")


add_executable(vc-gpu ${maxmatch_SOURCES} ${edmonds_CUDA_SOURCES} ${mis_SOURCES} ${vc-gpu_CUDA_SOURCES} ${maxmatch_CUDA_SOURCES} ${mis_CUDA_SOURCES} ${vc-gpu_SOURCES} ${edmonds_SOURCES})
target_include_directories(vc-gpu PRIVATE ${vc-gpu_INCLUDE_DIRS} ${pppcsr_INCLUDE_DIRS})
target_include_directories(vc-gpu SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS} "${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc/")
set_target_properties(vc-gpu PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
OUTPUT_NAME vc
CUDA_ARCHITECTURES "35;60;70;80")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
message("-- Debug build type detected, GPU_NPT setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
set_property(TARGET vc-gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(vc-gpu PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
else()
message("-- Release build type detected, -maxrregcount=31;-O3;-w; -gencode arch=compute_70,code=sm_70;")
target_compile_options(vc-gpu PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-maxrregcount=31;-O3;-w;-std=c++11>")
endif()
#target_link_libraries(vc-gpu parallel-packed-csr cuda)
