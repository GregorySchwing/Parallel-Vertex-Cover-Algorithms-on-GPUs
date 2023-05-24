
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"
#include <cooperative_groups.h>
using namespace cooperative_groups; 
#define VERTICES_PER_BLOCK 1000



#if USE_GLOBAL_MEMORY
struct GlobalDFSKernelArgs
{
    Stacks stacks; 
    unsigned int * minimum; 
    WorkList workList; 
    CSRGraph graph; 
    Counters* counters; 
    int* first_to_dequeue_global; 
    int* global_memory; 
    int* NODES_PER_SM;
};
#else
struct SharedDFSKernelArgs
{
    Stacks stacks; 
    unsigned int * minimum; 
    WorkList workList; 
    CSRGraph graph; 
    Counters* counters; 
    int* first_to_dequeue_global; 
    int* NODES_PER_SM;
};
#endif

/*
#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_DFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_DFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM) {
#endif
*/

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_DFS_kernel(GlobalDFSKernelArgs args) {
    Stacks stacks = args.stacks;
    unsigned int * minimum = args.minimum;
    WorkList workList = args.workList;
    CSRGraph graph = args.graph;
    Counters* counters = args.counters;
    int* first_to_dequeue_global = args.first_to_dequeue_global;
    int* global_memory = args.global_memory;
    int* NODES_PER_SM = args.NODES_PER_SM;
#else
__global__ void GlobalWorkList_shared_DFS_kernel(SharedDFSKernelArgs args) {
    Stacks stacks = args.stacks;
    unsigned int * minimum = args.minimum;
    WorkList workList = args.workList;
    CSRGraph graph = args.graph;
    Counters* counters = args.counters;
    int* first_to_dequeue_global = args.first_to_dequeue_global;
    int* NODES_PER_SM = args.NODES_PER_SM;
#endif

    __shared__ Counters blockCounters;
    initializeCounters(&blockCounters);
    //if (threadIdx.x==0)
    //printf("BlockID active %d\n", blockIdx.x);
    cooperative_groups::grid_group g = cooperative_groups::this_grid(); 
    #if USE_COUNTERS
        __shared__ unsigned int sm_id;
        if (threadIdx.x==0){
            sm_id=get_smid();
        }
    #endif

    int stackTop = -1;
    unsigned int stackSize = (stacks.minimum + 1);
    volatile int * stackVertexDegrees = &stacks.stacks[blockIdx.x * stackSize * graph.vertexNum];
    volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];

    // Define the vertexDegree_s
    unsigned int numDeletedVertices;
    unsigned int numDeletedVertices2;
    

    // vertexDegrees_s == stack1
    // vertexDegrees_s2 == stack2

    // numDeletedVertices == stack1 top
    // numDeletedVertices2 == stack2 top

    #if USE_GLOBAL_MEMORY
    int * vertexDegrees_s = &global_memory[graph.vertexNum*(2*blockIdx.x)];
    int * vertexDegrees_s2 = &global_memory[graph.vertexNum*(2*blockIdx.x + 1)];
    #else
    extern __shared__ int shared_mem[];
    int * vertexDegrees_s = shared_mem;
    int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];
    #endif
     __shared__ int bridgeFront;
    // Arrays: stack1, stack2, color, childsInDDFSTree, ddfsPredecessorsPtr, support
    // Scalars: stack1Top, stack2Top, globalColorCounter, supportTop, 
    if (threadIdx.x==0){
        bridgeFront = atomicAdd(graph.bridgeFront, 1);
        uint64_t edgePair = graph.bridgeList[bridgeFront];
    }
    __syncthreads();
    if(graph.removed[graph.bud[vertex]] || graph.removed[graph.bud[graph.dst[edgeIndex]]])
        continue;   
}
