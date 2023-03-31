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
#include "mis_kernels.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

extern "C"{
                 #include "bfstdcsc.h"

}

/* 
 * Function to compute a gpu-based parallel maximal matching for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format.
 *  
 */
//int  mm_gpu_csc (int *IC_h,int *CP_h,int *m_h,int *_m_d,int *req_h,int *c_h,int nz,int n,int repetition, int exec_protocol){
int mis_gpu(struct Graph * graph,struct MIS * mis, int exec_protocol){
  
  float t_mis;
  float t_mis_t = 0.0;
  int i,dimGrid;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *L_h=mis->L_h;

  int *CP_d = graph->graph_device.CP_d;
  int *IC_d = graph->graph_device.IC_d;
  int *L_d = mis->mis_device.L_d;
  int *c = mis->mis_device.c;
  int n = graph->N;
  int nz = graph->nz;
  int repetition = graph->repet;

  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  thrust::device_ptr<int> L_vec=thrust::device_pointer_cast(L_d);
  int L_size, L_sum=0;
  for (i = 0; i<repetition; i++){
    *c = 1;
    checkCudaErrors(cudaMemset(L_d,0,sizeof(*L_d)*graph->N));
    while(*c){
      *c = 0;
      cudaEventRecord(start);
      set_L<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d, IC_d, L_d, c, graph->N);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_mis,start,stop);
      t_mis_t += t_mis;
    }
    L_size = thrust::count(L_vec, L_vec+graph->N, 1);
    L_sum += L_size;
  }
  if (graph->seq)
    checkCudaErrors(cudaMemcpy(L_h,L_d, graph->N*sizeof(*L_h),cudaMemcpyDeviceToHost));
  int print_t = 1;
  if (graph->p){
    printf("mis_gpu::total time = %lfms \n",t_mis_t);
    printf("mis_gpu::average size = %lf \n", (float)L_sum/(float)graph->repet);
    printf("mis_gpu::average time = %lfms \n",t_mis_t/graph->repet);
  }
  return 0;
}//end bfs_gpu_td_csc_sc
