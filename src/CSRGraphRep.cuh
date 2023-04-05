#include "CSRGraphRep.h"
//includes CUDA project
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
CSRGraph allocateGraph(CSRGraph graph){
    CSRGraph Graph;

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;
    checkCudaErrors(cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum));
    checkCudaErrors(cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1)));
    checkCudaErrors(cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*(graph.vertexNum)));
    checkCudaErrors(cudaMalloc((void**) &num_unmatched_vertices_d,sizeof(unsigned int)));

    Graph.vertexNum = graph.vertexNum;
    Graph.edgeNum = graph.edgeNum;
    //Graph.num_unmatched_vertices = graph.num_unmatched_vertices;
    Graph.num_unmatched_vertices = num_unmatched_vertices_d;
    Graph.dst = dst_d;
    Graph.srcPtr = srcPtr_d;
    Graph.degree = degree_d;
    Graph.matching = matching_d;
    Graph.unmatched_vertices = unmatched_vertices_d;
    checkCudaErrors(cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice));
    //cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice));
    //cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice));

    return Graph;
}

void cudaFreeGraph(CSRGraph graph){
    checkCudaErrors(cudaFree(graph.dst));
    checkCudaErrors(cudaFree(graph.srcPtr));
    checkCudaErrors(cudaFree(graph.degree));
    if (graph.matching != NULL)
    checkCudaErrors(cudaFree(graph.matching));
    if (graph.unmatched_vertices != NULL)
    checkCudaErrors(cudaFree(graph.unmatched_vertices));
}