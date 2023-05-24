
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
    int * stack1 = &graph.stack1[graph.vertexNum*(blockIdx.x)];
    int * stack2 = &graph.stack2[graph.vertexNum*(blockIdx.x)];
    int * color = &graph.color[graph.vertexNum*(blockIdx.x)];
    int * childsInDDFSTree_keys = &graph.childsInDDFSTree_keys[graph.vertexNum*(blockIdx.x)];
    int * ddfsPredecessorsPtr = &graph.ddfsPredecessorsPtr[graph.vertexNum*(blockIdx.x)];
    int * support = &graph.support[graph.vertexNum*(blockIdx.x)];
    uint64_t * childsInDDFSTree_values = &graph.childsInDDFSTree_values[graph.vertexNum*(blockIdx.x)];

    unsigned int * stack1Top = &graph.stack1Top[(blockIdx.x)];
    unsigned int * stack2Top = &graph.stack2Top[(blockIdx.x)];
    unsigned int * supportTop = &graph.supportTop[(blockIdx.x)];
    unsigned int * globalColorCounter = &graph.globalColorCounter[(blockIdx.x)];
    unsigned int * childsInDDFSTreeTop = &graph.childsInDDFSTreeTop[(blockIdx.x)];

    #else
    extern __shared__ int shared_mem[];
    int * vertexDegrees_s = shared_mem;
    int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];

    int * stack1 = &graph.stack1[graph.vertexNum*0];
    int * stack2 = &graph.stack2[graph.vertexNum*1];
    int * color = &graph.color[graph.vertexNum*2];
    int * childsInDDFSTree_keys = &graph.childsInDDFSTree_keys[graph.vertexNum*3];
    int * ddfsPredecessorsPtr = &graph.ddfsPredecessorsPtr[graph.vertexNum*4];
    int * support = &graph.support[graph.vertexNum*5];
    uint64_t * childsInDDFSTree_values = &graph.childsInDDFSTree_values[graph.vertexNum*6];

    unsigned int * stack1Top = &graph.stack1Top[(graph.vertexNum*8)];
    unsigned int * stack2Top = &graph.stack2Top[(graph.vertexNum*8)+1];
    unsigned int * supportTop = &graph.supportTop[(graph.vertexNum*8)+2];
    unsigned int * globalColorCounter = &graph.globalColorCounter[(graph.vertexNum*8)+3];
    unsigned int * childsInDDFSTreeTop = &graph.childsInDDFSTreeTop[(graph.vertexNum*8)+4];

    #endif
     int bridgeFront;
     __shared__ int src;
     __shared__ int dst;

    // Arrays: stack1, stack2, color, childsInDDFSTree, ddfsPredecessorsPtr, support
    // Scalars: stack1Top, stack2Top, globalColorCounter, supportTop, 
    if (threadIdx.x==0){
        bridgeFront = atomicAdd(graph.bridgeFront, 1);
        if (bridgeFront >= graph.bridgeList_counter[0]) return;
        src = (uint32_t)graph.bridgeList[bridgeFront];
        dst = (graph.bridgeList[bridgeFront] >> 32);
    //}
    //__syncthreads();
    printf("Bridge ID %d src %d dst %d\n", bridgeFront, src, dst);
    if(graph.removed[graph.bud[src]] || graph.removed[graph.bud[dst]]) return;   
    auto ddfsResult = ddfs(graph,src,dst,stack1,stack2,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,ddfsPredecessorsPtr,childsInDDFSTree_keys,childsInDDFSTree_values,childsInDDFSTreeTop);
    //printf("bridgeID %d bridge %d %d support %d %d %d %d %d %d\n", bridgeFront, src, dst, support[0], support[1], supportTop[0]>2 ? support[2]:-1,supportTop[0]>3 ? support[3]:-1,supportTop[0]>4 ? support[4]:-1,supportTop[0]>5 ? support[5]:-1);
    }
}
