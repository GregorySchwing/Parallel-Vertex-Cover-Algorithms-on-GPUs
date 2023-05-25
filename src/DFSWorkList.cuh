#ifndef DFSWL_H
#define DFSWL_H
#include "CSRGraphRep.h"
#include <cuda/std/utility>
#define st first
#define nd second
typedef cuda::std::pair<int,int> pii;

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


struct DFSWorkList{
    unsigned int vertexNum;
    unsigned int edgeNum;

    cuda::std::pair<pii,pii>* myBridge;
    pii* bridgeList;
    unsigned int *bridgeList_counter;
    void reset();
};

void DFSWorkList::reset(){
  cudaMemset(bridgeList_counter, 0, sizeof(unsigned int));
  cudaMemset(myBridge, 0, vertexNum*sizeof(cuda::std::pair<pii,pii>));

}

DFSWorkList allocateDFSWorkList(CSRGraph graph){
    DFSWorkList DFSWL;
    cuda::std::pair<pii,pii>* myBridge_d;
    pii* bridgeList_d;
    unsigned int *bridgeList_counter_d;

    checkCudaErrors(cudaMalloc((void**) &bridgeList_counter_d,sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &bridgeList_d,2*graph.edgeNum*sizeof(pii)));
    checkCudaErrors(cudaMalloc((void**) &myBridge_d,graph.vertexNum*sizeof(cuda::std::pair<pii,pii>)));

    cudaMemset(bridgeList_counter_d, 0, sizeof(unsigned int));
    DFSWL.vertexNum = graph.vertexNum;
    DFSWL.edgeNum = graph.edgeNum;

    DFSWL.bridgeList_counter = bridgeList_counter_d;
    DFSWL.bridgeList = bridgeList_d;
    DFSWL.myBridge=myBridge_d;

    return DFSWL;
}

void cudaFreeDFSWorkList(DFSWorkList dfsWL){
    cudaFree(dfsWL.bridgeList);
    cudaFree(dfsWL.bridgeList_counter);
}
#endif