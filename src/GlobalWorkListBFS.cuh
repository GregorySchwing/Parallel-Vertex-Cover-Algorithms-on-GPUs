
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"


__global__ void GlobalWorkList_Set_Sources_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM) {

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= graph.vertexNum) return;

    if (graph.matching[vertex]==-1){
        setlvl(graph,vertex,0);
        //printf("Block %d setting vertex %d as src oddlvl %d evenlvl %d\n", blockIdx.x,vertex,graph.oddlvl[vertex],graph.evenlvl[vertex]);
    }
}


__global__ void GlobalWorkList_BFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= graph.vertexNum) return;
    unsigned int start = graph.srcPtr[vertex];
    unsigned int end = graph.srcPtr[vertex + 1];
    unsigned int edgeIndex;
    for(edgeIndex=start; edgeIndex < end-start; edgeIndex++) { // Delete Neighbors of startingVertex
        if (graph.edgeStatus[edgeIndex] == NotScanned && (graph.oddlvl[vertex] == depth) == (graph.matching[vertex] == graph.dst[edgeIndex])) {
            if(min(graph.oddlvl[graph.dst[edgeIndex]], graph.evenlvl[graph.dst[edgeIndex]]) >= depth+1) {
                graph.edgeStatus[edgeIndex] = Prop;
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                //printf("BID %d prop edge %d - %d \n", blockIdx.x, vertex, graph.dst[edgeIndex]);
                for(edgeIndexOther=startOther; edgeIndexOther < endOther-startOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Prop;
                        break;
                    }
                }
                if(minlvl(graph,graph.dst[edgeIndex]) > depth+1) {
                    if(depth&1) graph.oddlvl[vertex] = depth; else graph.evenlvl[vertex] = depth;
                }
                graph.pred[edgeIndexOther]=true;
            }
            else{
                graph.edgeStatus[edgeIndex] = Bridge;
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                //printf("BID %d bridge edge %d - %d \n", blockIdx.x, vertex, graph.dst[edgeIndex]);
                for(edgeIndexOther=startOther; edgeIndexOther < endOther-startOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Bridge;
                        break;
                    }
                }
                if(tenacity(graph,vertex,graph.dst[edgeIndex]) < INF) {
                    graph.bridges[edgeIndex] = tenacity(graph,vertex,graph.dst[edgeIndex]);
                }
            }
        }
    }
}
