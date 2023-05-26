
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "DFSWorkList.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"


__global__ void GlobalWorkList_Set_Sources_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM) {

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= graph.vertexNum) return;

    if (graph.matching[vertex]==-1){
        setLvl(graph,vertex,0);
        printf("Block %d setting vertex %d as src oddlvl %d evenlvl %d\n", blockIdx.x,vertex,graph.oddlvl[vertex],graph.evenlvl[vertex]);
    } else {
        printf("Block %d not setting vertex %d as src oddlvl %d evenlvl %d\n", blockIdx.x,vertex,graph.oddlvl[vertex],graph.evenlvl[vertex]);
    }
}


__global__ void GlobalWorkList_BFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    unsigned int vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if(vertex >= graph.vertexNum || depth != getLvl(graph, vertex, depth)) return;

    unsigned int start = graph.srcPtr[vertex];
    unsigned int end = graph.srcPtr[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
        printf("src %d dst %d start %d end %d edgeIndex %d edgeStatus %d evenlvl %d oddlvl %d matching %d\n",vertex,graph.dst[edgeIndex],start,end,edgeIndex,graph.edgeStatus[edgeIndex],graph.oddlvl[vertex],graph.evenlvl[vertex],graph.matching[vertex]);
        if (graph.edgeStatus[edgeIndex] == NotScanned && (graph.oddlvl[vertex] == depth) == (graph.matching[vertex] == graph.dst[edgeIndex])) {
            if(minlvl(graph,graph.dst[edgeIndex]) >= depth+1) {
                graph.edgeStatus[edgeIndex] = Prop;
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                printf("BID %d prop edge %d - %d \n", blockIdx.x, vertex, graph.dst[edgeIndex]);
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Prop;
                        break;
                    }
                }
                if(minlvl(graph,graph.dst[edgeIndex]) > depth+1)
                    setLvl(graph, graph.dst[edgeIndex], depth+1);
                graph.pred[edgeIndexOther]=true;
            }
            else{
                graph.edgeStatus[edgeIndex] = Bridge;
                printf("BID %d bridge edge %d - %d tenacity %d \n", blockIdx.x, vertex,graph.dst[edgeIndex],tenacity(graph,vertex,graph.dst[edgeIndex]));
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Bridge;
                        break;
                    }
                }
                if(tenacity(graph,vertex,graph.dst[edgeIndex]) < INF) {
                    graph.bridgeTenacity[edgeIndex] = tenacity(graph,vertex,graph.dst[edgeIndex]);
                }
            }
        }
    }
}

__global__ void GlobalWorkList_Extract_Bridges_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, DFSWorkList dfsWL, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    if(vertex >= graph.vertexNum) return;
    unsigned int start = graph.srcPtr[vertex];
    unsigned int end = graph.srcPtr[vertex + 1];
    unsigned int edgeIndex = start;
    for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
        if (graph.edgeStatus[edgeIndex] == Bridge && graph.bridgeTenacity[edgeIndex] == 2*depth+1) {
            unsigned int top = atomicAdd(dfsWL.bridgeList_counter,1);
            //uint64_t edgePair = (uint64_t) vertex << 32 | graph.dst[edgeIndex];
            //printf("Adding bridge %d %d\n", vertex, graph.dst[edgeIndex]);
            dfsWL.bridgeList[top] = cuda::std::pair<int,int>(vertex, graph.dst[edgeIndex]);
        }
    }
}



__global__ void GlobalWorkList_BFS_kernel_st(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    unsigned int vertex;
    for (vertex = 0; vertex < graph.vertexNum; ++vertex){
    if(vertex >= graph.vertexNum || depth != getLvl(graph, vertex, depth)) continue;

    unsigned int start = graph.srcPtr[vertex];
    unsigned int end = graph.srcPtr[vertex + 1];
    unsigned int edgeIndex=start;
    for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
        printf("src %d dst %d start %d end %d edgeIndex %d edgeStatus %d evenlvl %d oddlvl %d matching %d\n",vertex,graph.dst[edgeIndex],start,end,edgeIndex,graph.edgeStatus[edgeIndex],graph.oddlvl[vertex],graph.evenlvl[vertex],graph.matching[vertex]);
        if (graph.edgeStatus[edgeIndex] == NotScanned && (graph.oddlvl[vertex] == depth) == (graph.matching[vertex] == graph.dst[edgeIndex])) {
            if(minlvl(graph,graph.dst[edgeIndex]) >= depth+1) {
                graph.edgeStatus[edgeIndex] = Prop;
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                printf("BID %d prop edge %d - %d \n", blockIdx.x, vertex, graph.dst[edgeIndex]);
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Prop;
                        break;
                    }
                }
                if(minlvl(graph,graph.dst[edgeIndex]) > depth+1)
                    setLvl(graph, graph.dst[edgeIndex], depth+1);
                graph.pred[edgeIndexOther]=true;
            }
            else{
                graph.edgeStatus[edgeIndex] = Bridge;
                printf("BID %d bridge edge %d - %d tenacity %d \n", blockIdx.x, vertex,graph.dst[edgeIndex],tenacity(graph,vertex,graph.dst[edgeIndex]));
                unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                unsigned int edgeIndexOther;
                for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) { // Delete Neighbors of startingVertex
                    if(vertex==graph.dst[edgeIndexOther]){
                        graph.edgeStatus[edgeIndexOther] = Bridge;
                        break;
                    }
                }
                if(tenacity(graph,vertex,graph.dst[edgeIndex]) < INF) {
                    graph.bridgeTenacity[edgeIndex] = tenacity(graph,vertex,graph.dst[edgeIndex]);
                }
            }
        }
    }
    }
}

__global__ void GlobalWorkList_Extract_Bridges_kernel_st(Stacks stacks, unsigned int * minimum, WorkList workList, DFSWorkList dfsWL, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    unsigned int vertex = threadIdx.x + blockIdx.x*(blockDim.x);
    vertex = 0;
    for (vertex = 0; vertex < graph.vertexNum; ++vertex){
        unsigned int start = graph.srcPtr[vertex];
        unsigned int end = graph.srcPtr[vertex + 1];
        unsigned int edgeIndex = start;
        for(; edgeIndex < end; edgeIndex++) { // Delete Neighbors of startingVertex
            if (graph.edgeStatus[edgeIndex] == Bridge && graph.bridgeTenacity[edgeIndex] == 2*depth+1) {
                unsigned int top = atomicAdd(dfsWL.bridgeList_counter,1);
                //uint64_t edgePair = (uint64_t) vertex << 32 | graph.dst[edgeIndex];
                //printf("Adding bridge %d %d\n", vertex, graph.dst[edgeIndex]);
                dfsWL.bridgeList[top] = cuda::std::pair<int,int>(vertex, graph.dst[edgeIndex]);
            }
        }
    }
}


__global__ void PrintTops(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM, unsigned int depth) {

    if (threadIdx.x==0){
    printf("BlockID %d stack1Top %d stack2Top %d\n", blockIdx.x, graph.stack1Top[blockIdx.x], graph.stack2Top[blockIdx.x]);
    }
}
