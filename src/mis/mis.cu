/*
 * This source code is distributed under the terms defined  
 * in the file bfstdcsc_main.c of this source distribution.
 */
/* 
 *  Breadth first search (BFS) 
 *  Single precision (float data type) 
 *  TurboBFS_CSC_TD:bfsgputdcsc_sc.cu
 * 
 *  This program computes the GPU-based parallel 
 *  top-down BFS (scalar) for unweighted graphs represented 
 *  by sparse adjacency matrices in the CSC format, including
 *  the computation of the S array to store the depth at 
 *  which each vertex is discovered.  
 *
 */

#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>

//includes CUDA project
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
extern "C"{
                 #include "bfstdcsc.h"

}


MIS_GPU create_mis_gpu_struct(int N){
  MIS_GPU mis_device;

  /*Allocate device memory for the vector m_d */
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&mis_device.L_d),sizeof(*mis_device.L_d)*(N)));

  /*allocate unified memory for integer variable c for control of while loop*/
  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&mis_device.c),sizeof(*mis_device.c)));

  return mis_device;
};