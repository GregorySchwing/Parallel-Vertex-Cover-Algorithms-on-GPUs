#include "CSRGraphRep.h"

CSRGraph allocateGraph(CSRGraph graph){
    CSRGraph Graph;

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    int num_unmatched_vertices;
    unsigned int *unmatched_vertices_d;
    cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum);
    cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1));
    cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*((graph.vertexNum/2)+1));

    Graph.vertexNum = graph.vertexNum;
    Graph.edgeNum = graph.edgeNum;
    Graph.num_unmatched_vertices = graph.num_unmatched_vertices;

    Graph.dst = dst_d;
    Graph.srcPtr = srcPtr_d;
    Graph.degree = degree_d;
    Graph.matching = matching_d;
    Graph.unmatched_vertices = unmatched_vertices_d;
    cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice);
    cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice);
    cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);
    cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice);
    cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);

    return Graph;
}

void cudaFreeGraph(CSRGraph graph){
    cudaFree(graph.dst);
    cudaFree(graph.srcPtr);
    cudaFree(graph.degree);
    if (graph.matching != NULL)
    cudaFree(graph.matching);
    if (graph.unmatched_vertices != NULL)
    cudaFree(graph.unmatched_vertices);
}