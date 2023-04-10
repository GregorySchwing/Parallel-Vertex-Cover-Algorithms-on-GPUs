#include "CSRGraphRep.h"
#include "ThrustGraph.h"

void createfromThrustGraph(CSRGraph &Graph, ThrustGraph & tg){

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;
    //cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum);
    //cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1));
    //cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum);

    Graph.vertexNum = tg.vertexNum;
    Graph.edgeNum = tg.edgeNum;
    //Graph.num_unmatched_vertices = graph.num_unmatched_vertices;
    Graph.srcPtr = thrust::raw_pointer_cast(tg.offsets_h.data() );
    Graph.dst = thrust::raw_pointer_cast( tg.values_h.data() );
    Graph.degree = thrust::raw_pointer_cast( tg.degrees_h.data() );

    //cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice);
    //cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice);
    //cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);

}


CSRGraph createGPUfromThrustGraph(CSRGraph &Graph, ThrustGraph & tg){
    

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;

    Graph.vertexNum = tg.vertexNum;
    Graph.edgeNum = tg.edgeNum;

    Graph.srcPtr = thrust::raw_pointer_cast(tg.offsets_d.data() );
    Graph.dst = thrust::raw_pointer_cast( tg.values_d.data() );
    Graph.degree = thrust::raw_pointer_cast( tg.degrees_d.data() );

    //cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice);
    //cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice);
    //cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);

    return;
}

CSRGraph allocateGraph(CSRGraph graph){
    CSRGraph Graph;

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;
    cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum);
    cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1));
    cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*(graph.vertexNum));
    cudaMalloc((void**) &num_unmatched_vertices_d,sizeof(unsigned int));

    Graph.vertexNum = graph.vertexNum;
    Graph.edgeNum = graph.edgeNum;
    //Graph.num_unmatched_vertices = graph.num_unmatched_vertices;
    Graph.num_unmatched_vertices = num_unmatched_vertices_d;
    Graph.dst = dst_d;
    Graph.srcPtr = srcPtr_d;
    Graph.degree = degree_d;
    Graph.matching = matching_d;
    Graph.unmatched_vertices = unmatched_vertices_d;
    cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice);
    cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice);
    cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice);
    //cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);

    return Graph;
}


CSRGraph allocateGraph2(CSRGraph graph){
    CSRGraph Graph;

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;

    cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*(graph.vertexNum));
    cudaMalloc((void**) &num_unmatched_vertices_d,sizeof(unsigned int));

    Graph.vertexNum = graph.vertexNum;
    Graph.edgeNum = graph.edgeNum;
    //Graph.num_unmatched_vertices = graph.num_unmatched_vertices;
    Graph.num_unmatched_vertices = num_unmatched_vertices_d;

    Graph.matching = matching_d;
    Graph.unmatched_vertices = unmatched_vertices_d;

    return Graph;
}

void cudaFreeGraph(CSRGraph graph){
    //cudaFree(graph.dst);
    //cudaFree(graph.srcPtr);
    //cudaFree(graph.degree);
    if (graph.matching != NULL)
    cudaFree(graph.matching);
    if (graph.unmatched_vertices != NULL)
    cudaFree(graph.unmatched_vertices);
}