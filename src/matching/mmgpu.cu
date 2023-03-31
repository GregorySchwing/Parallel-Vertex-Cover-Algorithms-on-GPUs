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
#include "matchgpu.h"
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

struct is_less_than_0
{
  __host__ __device__
  int operator()(int &x)
  {
    if (x > -1)
      return x;
    else
      return -1;
  }
};

/* 
 * Function to compute a gpu-based parallel maximal matching for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format.
 *  
 */
//int  mm_gpu_csc (int *IC_h,int *CP_h,int *m_h,int *_m_d,int *req_h,int *c_h,int nz,int n,int repetition, int exec_protocol){
int mm_gpu_csc(struct Graph * graph,struct match * m, int exec_protocol){
//int  mm_gpu_csc (int *IC_h,int *CP_h,int *m_h,int *_m_d,int *req_h,int *c_h,int nz,int n,int repetition, int exec_protocol){
  float t_mm;
  float t_mm_t = 0.0;
  float t_thrust;
  float t_thrust_t = 0.0;
  int i,dimGrid;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int *CP_h = graph->CscA.CP;
  int *IC_h = graph->CscA.IC;
  int *m_h = m->m_h;
  int *m_hgpu = m->m_hgpu;
  int n = graph->N;
  int nz = graph->nz;
  int repetition = graph->repet;

  int *CP_d;
  int *IC_d;
  int *m_d;
  int *req_d;
  int *c;
  int result, resultSum = 0;


  if (exec_protocol){
    CP_d = graph->graph_device.CP_d;
    IC_d = graph->graph_device.IC_d;
    m_d = m->match_device.m_d;
    req_d = m->match_device.req_d;
    c = m->match_device.c;
  } else {
    /*Allocate device memory for the vector CP_d */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&CP_d),sizeof(*CP_d)*(n+1)));
    /*Copy host memory (CP_h) to device memory (CP_d)*/
    checkCudaErrors(cudaMemcpy(CP_d,CP_h,(n+1)*sizeof(*CP_d),cudaMemcpyHostToDevice));

    /*Allocate device memory for the vector IC_d */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&IC_d),sizeof(*IC_d)*nz));
    /*Copy host memory (IC_h) to device memory (IC_d)*/
    checkCudaErrors(cudaMemcpy(IC_d,IC_h,nz*sizeof(*IC_d),cudaMemcpyHostToDevice));

    /*Allocate device memory for the vector m_d, and set m_d to zero. */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&m_d),sizeof(*m_d)*n));
    checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));

    /*Allocate device memory for the vector f_d*/
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&req_d),sizeof(*req_d)*n));

    /*allocate unified memory for integer variable c for control of while loop*/
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&c),sizeof(*c)));
  }


  srand(1);
  /*computing MM */
  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  for (i = 0; i<repetition; i++){
    *c = 1;
    //d = 0;
    checkCudaErrors(cudaMemset(req_d,0,sizeof(*req_d)*n));
    checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));
    int count = 0;
    while (*c && ++count < NR_MAX_MATCH_ROUNDS){
      //d = d + 1;
      *c = 0;
      cudaEventRecord(start);
      gaSelect<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, c, n, rand());
      grRequest<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      grRespond<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      gMatch<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, req_d, n);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_mm,start,stop);
      t_mm_t += t_mm;
    }
    cudaEventRecord(start);
    using namespace thrust::placeholders;
    thrust::device_ptr<int> m_vec=thrust::device_pointer_cast(m_d);
    thrust::for_each(m_vec, m_vec+n, _1 -= 4);
    
    result = thrust::count(m_vec, m_vec+n, -1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_thrust,start,stop);
    t_thrust_t += t_thrust;
    resultSum += result;

  }
  printf("\nbfs_gpu_mm_csc_sc::t_sum=%lfms \n",t_mm_t+t_thrust_t);
  
  int print_t = 1;
  if (print_t){
    printf("mm_gpu_csc::average time mm = %lfms, avg matched %f, avg unmatched %f, total %d\n",(t_mm_t+t_thrust_t)/repetition, 
      (float)n-((float)resultSum/(float)repetition), ((float)resultSum/(float)repetition), n);
  }
  /*cleanup memory*/
  if (exec_protocol){
    // Really only necessary if checking against seq malt bfs
    checkCudaErrors(cudaMemcpy(m_h,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hgpu,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));

  } else {
    /*Copy device memory (m_d) to host memory (S_h)*/
    checkCudaErrors(cudaMemcpy(m_h,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(CP_d));
    checkCudaErrors(cudaFree(IC_d));
    checkCudaErrors(cudaFree(m_d));
    checkCudaErrors(cudaFree(req_d));
    checkCudaErrors(cudaFree(c));
  }
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return 0;
}//end bfs_gpu_td_csc_sc

int mm_gpu_csc_from_mis(struct Graph * graph,struct match * m,struct MIS * mis, int exec_protocol){
//int  mm_gpu_csc (int *IC_h,int *CP_h,int *m_h,int *_m_d,int *req_h,int *c_h,int nz,int n,int repetition, int exec_protocol){
  float t_mm;
  float t_mm_t = 0.0;
  float t_thrust;
  float t_thrust_t = 0.0;
  float t_mis;
  float t_mis_t = 0.0;
  int i,dimGrid;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int *CP_h = graph->CscA.CP;
  int *IC_h = graph->CscA.IC;
  int *m_h = m->m_h;
  int *m_hgpu = m->m_hgpu;
  int n = graph->N;
  int nz = graph->nz;
  int repetition = graph->repet;

  int *CP_d;
  int *IC_d;
  int *m_d;
  int *req_d;
  int *L_d;
  int *c;
  int result, resultSum = 0;

  if (exec_protocol){
    CP_d = graph->graph_device.CP_d;
    IC_d = graph->graph_device.IC_d;
    m_d = m->match_device.m_d;
    req_d = m->match_device.req_d;
    L_d = mis->mis_device.L_d;
    c = m->match_device.c;
  } else {
    /*Allocate device memory for the vector CP_d */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&CP_d),sizeof(*CP_d)*(n+1)));
    /*Copy host memory (CP_h) to device memory (CP_d)*/
    checkCudaErrors(cudaMemcpy(CP_d,CP_h,(n+1)*sizeof(*CP_d),cudaMemcpyHostToDevice));

    /*Allocate device memory for the vector IC_d */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&IC_d),sizeof(*IC_d)*nz));
    /*Copy host memory (IC_h) to device memory (IC_d)*/
    checkCudaErrors(cudaMemcpy(IC_d,IC_h,nz*sizeof(*IC_d),cudaMemcpyHostToDevice));

    /*Allocate device memory for the vector m_d, and set m_d to zero. */
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&m_d),sizeof(*m_d)*n));
    checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));

    /*Allocate device memory for the vector f_d*/
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&req_d),sizeof(*req_d)*n));

    /*allocate unified memory for integer variable c for control of while loop*/
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&c),sizeof(*c)));
  }

  using namespace thrust::placeholders;
  thrust::device_ptr<int> m_vec=thrust::device_pointer_cast(m_d);

  srand(1);
  /*computing MM */
  dimGrid = (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
  thrust::device_ptr<int> L_vec=thrust::device_pointer_cast(L_d);
  int L_size, L_sum=0;
  for (i = 0; i<repetition; i++){
    *c = 1;
    //d = 0;
    checkCudaErrors(cudaMemset(req_d,0,sizeof(*req_d)*n));
    checkCudaErrors(cudaMemset(m_d,0,sizeof(*m_d)*n));

    *c = 1;
    int lastMatch =0;
    result = 0;
    do {
      lastMatch = result;
      checkCudaErrors(cudaMemset(L_d,0,sizeof(*L_d)*graph->N));
      while(*c){
        *c = 0;
        cudaEventRecord(start);
        set_L_unmatched<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d, IC_d, L_d, m_d,c, graph->N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_mis,start,stop);
        t_mis_t += t_mis;
      }
      L_size = thrust::count(L_vec, L_vec+graph->N, 1);
      //printf("\nMIS count::t_sum=%dms \n",L_size);

      L_sum += L_size;

      int count = 0;
      *c = 1;
      while (*c && ++count < NR_MAX_MATCH_ROUNDS){
        //d = d + 1;
        *c = 0;
        cudaEventRecord(start);
        gaSelect_from_mis<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, c, L_d,n);
        grRequest<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
        grRespond<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
        gMatch<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, req_d, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_mm,start,stop);
        t_mm_t += t_mm;
      }
      result = thrust::count_if(m_vec, m_vec+n, _1>3);
      //printf("\nmatch count::t_sum=%dms \n",result);
    }while(result!=lastMatch);
    int count = 0;
    *c = 1;
    while (*c && ++count < NR_MAX_MATCH_ROUNDS){
      //d = d + 1;
      *c = 0;
      cudaEventRecord(start);
      gaSelect<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, c,n,rand());
      grRequest<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      grRespond<<<dimGrid,THREADS_PER_BLOCK>>>(CP_d,IC_d,req_d, m_d, n);
      gMatch<<<dimGrid,THREADS_PER_BLOCK>>>(m_d, req_d, n);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t_mm,start,stop);
      t_mm_t += t_mm;
    }
    cudaEventRecord(start);
    using namespace thrust::placeholders;
    thrust::device_ptr<int> m_vec=thrust::device_pointer_cast(m_d);
    thrust::for_each(m_vec, m_vec+n, _1 -= 4);
    //thrust::for_each(m_vec, m_vec+n, is_less_than_0());

    result = thrust::count_if(m_vec, m_vec+n, _1<0);
    int badresult = thrust::count_if(m_vec, m_vec+n, _1<-1);
    if (badresult){
      printf ("Bad results %d\n", badresult);
      exit(1);
    }
    //result = thrust::count(m_vec, m_vec+n, -1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_thrust,start,stop);
    t_thrust_t += t_thrust;
    resultSum += result;
    
  }
  printf("\nmm_gpu_csc_from_mis::t_sum=%lfms \n",t_mm_t+t_thrust_t);
  
  int print_t = 1;
  if (print_t){
    printf("mm_gpu_csc_from_mis::average time mm = %lfms, avg matched %f, avg unmatched %f, total %d\n",(t_mm_t+t_thrust_t)/repetition, 
      (float)n-((float)resultSum/(float)repetition), ((float)resultSum/(float)repetition), n);
  }
  /*cleanup memory*/
  if (exec_protocol){
    // Really only necessary if checking against seq malt bfs
    checkCudaErrors(cudaMemcpy(m_h,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hgpu,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));

  } else {
    /*Copy device memory (m_d) to host memory (S_h)*/
    checkCudaErrors(cudaMemcpy(m_h,m_d, n*sizeof(*m_d),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(CP_d));
    checkCudaErrors(cudaFree(IC_d));
    checkCudaErrors(cudaFree(m_d));
    checkCudaErrors(cudaFree(req_d));
    checkCudaErrors(cudaFree(c));
  }
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return 0;
}//end bfs_gpu_td_csc_sc



/* 
 * Function to compute a gpu-based parallel maximal matching for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format.
 *  
 */
void add_edges_to_unmatched_from_last_vertex_gpu_csc(struct Graph * graph,struct match * m,int exec_protocol){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Local copies are only used when exec_protocol is 0
  // This is turning into a debug mode execution.
  /*Allocate device memory for the vector CP_d */
  int *CP_d;
  /*Allocate device memory for the vector IC_d */
  int *IC_d;
  /*Allocate device memory for the vector m_d, and set m_d to zero. */
  int *m_d;

  if (exec_protocol){
    CP_d = graph->graph_device.CP_d;
    IC_d = graph->graph_device.IC_d;
    m_d = m->match_device.m_d;
  } else {
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&CP_d),sizeof(*CP_d)*(graph->N_including_supersource+1)));
    /*Copy host memory (CP_h) to device memory (CP_d)*/
    checkCudaErrors(cudaMemcpy(CP_d,graph->CscA.CP,(graph->N+1)*sizeof(*CP_d),cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&IC_d),sizeof(*IC_d)*graph->nz_including_supersource));
    /*Copy host memory (IC_h) to device memory (IC_d)*/
    checkCudaErrors(cudaMemcpy(IC_d,graph->CscA.IC,graph->nz*sizeof(*IC_d),cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&m_d),sizeof(*m_d)*graph->N));
    checkCudaErrors(cudaMemcpy(m_d,m->m_h,graph->N*sizeof(*m_d),cudaMemcpyHostToDevice));
  }

  thrust::device_ptr<int> m_vec=thrust::device_pointer_cast(m_d);

  using namespace thrust;
  using namespace thrust::placeholders;

 // storage for the nonzero indices
 thrust::device_vector<int> indices(graph->N);
 // compute indices of nonzero elements
 typedef thrust::device_vector<int>::iterator IndexIterator;
 IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
                                             thrust::make_counting_iterator(graph->N),
                                             m_vec,
                                             indices.begin(),
                                             _1 == -1);

  printf("\nnum unmatched %d\n", indices_end-indices.begin());
  m->num_matched_h[0] = graph->N-(indices_end-indices.begin());
  // Set number of sources
  thrust::device_ptr<int> CP_vec=thrust::device_pointer_cast(CP_d);

  // NumSources = (indices_end-indices.begin())
  // Previous index = nz
  // Prefix sum = previous index + NumSources
  thrust::device_vector<int> numSourcesPrefixSum(1);

  numSourcesPrefixSum[0] = graph->nz+(indices_end-indices.begin());

  // CP is int[N+2], N_including_supersource = N+1, CP[N+1]=num unmatched+nz
  thrust::copy(thrust::device, numSourcesPrefixSum.begin(), numSourcesPrefixSum.end(), CP_vec+graph->N_including_supersource);

  // Set sources
  thrust::device_ptr<int> IC_vec=thrust::device_pointer_cast(IC_d);

  // IC is int[nz+((N+1)/2)], 
  // IC[nz]...IC[nz+num unmatched] = sources
  //printf("Max edge count %d\n", nz_including_supersource);
  //printf("Curr edge count %d\n", nz);
  //printf("New edge count %d\n", nz+(indices_end-indices.begin()));

  // Copy sources into column array IC at CP[N] to CP[N+1]
  thrust::copy(thrust::device, indices.begin(), indices_end, IC_vec+graph->nz);

  if (graph->seq){
    checkCudaErrors(cudaMemcpy(graph->CscA_hgpu.CP,CP_d,(graph->N_including_supersource+1)*sizeof(*CP_d),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(graph->CscA_hgpu.IC,IC_d,graph->nz_including_supersource*sizeof(*IC_d),cudaMemcpyDeviceToHost));

  }

  if (exec_protocol){

  } else {
    checkCudaErrors(cudaFree(CP_d));
    checkCudaErrors(cudaFree(IC_d));
    checkCudaErrors(cudaFree(m_d));
  }
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return;
}//end bfs_gpu_td_csc_sc