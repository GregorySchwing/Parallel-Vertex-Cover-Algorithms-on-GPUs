
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
    int * budAtDDFSEncounter = &graph.budAtDDFSEncounter[2*graph.edgeNum*(blockIdx.x)];
    int * ddfsPredecessorsPtr = &graph.ddfsPredecessorsPtr[graph.vertexNum*(blockIdx.x)];
    int * support = &graph.support[graph.vertexNum*(blockIdx.x)];
    int * removedVerticesQueue = &graph.removedVerticesQueue[graph.vertexNum*(blockIdx.x)];
    int * removedPredecessorsSize = &graph.removedVerticesQueue[graph.vertexNum*(blockIdx.x)];

    unsigned int * stack1Top = &graph.stack1Top[(blockIdx.x)];
    unsigned int * stack2Top = &graph.stack2Top[(blockIdx.x)];
    unsigned int * supportTop = &graph.supportTop[(blockIdx.x)];
    unsigned int * globalColorCounter = &graph.globalColorCounter[(blockIdx.x)];
    unsigned int * removedVerticesQueueBack = &graph.removedVerticesQueueBack[(blockIdx.x)];    
    unsigned int * removedVerticesQueueFront = &graph.removedVerticesQueueFront[(blockIdx.x)];    

    bool * foundPath = &graph.foundPath[(blockIdx.x)];

    #else
    extern __shared__ int shared_mem[];
    int * vertexDegrees_s = shared_mem;
    int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];

    int * stack1 = &graph.stack1[graph.vertexNum*0];
    int * stack2 = &graph.stack2[graph.vertexNum*1];
    int * color = &graph.color[graph.vertexNum*2];
    int * ddfsPredecessorsPtr = &graph.ddfsPredecessorsPtr[graph.vertexNum*3];
    int * support = &graph.support[graph.vertexNum*4];
    int * removedVerticesQueue = &graph.removedVerticesQueue[graph.vertexNum*5];
    int * removedPredecessorsSize = &graph.removedVerticesQueue[graph.vertexNum*6];
    int * budAtDDFSEncounter = &graph.budAtDDFSEncounter[graph.vertexNum*7];

    unsigned int * stack1Top = &graph.stack1Top[(graph.vertexNum*7)+(2*graph.edgeNum)];
    unsigned int * stack2Top = &graph.stack2Top[(graph.vertexNum*7)+(2*graph.edgeNum)+1];
    unsigned int * supportTop = &graph.supportTop[(graph.vertexNum*7)+(2*graph.edgeNum)+2];
    unsigned int * globalColorCounter = &graph.globalColorCounter[(graph.vertexNum*7)+(2*graph.edgeNum)+3];
    unsigned int * removedVerticesQueueBack = &graph.removedVerticesQueueBack[(graph.vertexNum*7)+(2*graph.edgeNum)+4];
    unsigned int * removedVerticesQueueFront = &graph.removedVerticesQueueFront[(graph.vertexNum*7)+(2*graph.edgeNum)+5];
    bool * foundPath = &graph.foundPath[(graph.vertexNum*7)+(2*graph.edgeNum)+6];

    #endif
     int bridgeFront;
     __shared__ int src;
     __shared__ int dst;
     __shared__ uint64_t ddfsResult;

    // Arrays: stack1, stack2, color, childsInDDFSTree, ddfsPredecessorsPtr, support, myBridge
    // Scalars: stack1Top, stack2Top, globalColorCounter, supportTop, 
    if (threadIdx.x==0){
        bridgeFront = atomicAdd(graph.bridgeFront, 1);
        if (bridgeFront >= graph.bridgeList_counter[0]) return;
        src = (uint32_t)graph.bridgeList[bridgeFront];
        dst = (graph.bridgeList[bridgeFront] >> 32);

        printf("Bridge ID %d src %d dst %d\n", bridgeFront, src, dst);
        if(graph.removed[graph.bud[src]] || graph.removed[graph.bud[dst]]) return;   
        ddfsResult = ddfs(graph,src,dst,stack1,stack2,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,ddfsPredecessorsPtr,budAtDDFSEncounter);
        printf("bridgeID %d bridge %d %d support %d %d %d %d %d %d\n", bridgeFront, src, dst, support[0], support[1], supportTop[0]>2 ? support[2]:-1,supportTop[0]>3 ? support[3]:-1,supportTop[0]>4 ? support[4]:-1,supportTop[0]>5 ? support[5]:-1);

        //pair<pii,pii> curBridge = {b,{bud[b.st], bud[b.nd]}};
        __int128_t curBridge =  (__int128_t) src                << 96 |
                                (__int128_t) dst                << 64 |
                                (__int128_t) graph.bud[src]     << 32 |
                                             graph.bud[dst];

        /*

        for(auto v:support) {
            if(v == ddfsResult.second) continue; //skip bud
            myBridge[v] = curBridge;
            bud.linkTo(v,ddfsResult.second);

            //this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths 
            setLvl(v,2*i+1-minlvl(v));
            for(auto f : graph[v])
                if(evenlvl[v] > oddlvl[v] && f.type == Bridge && tenacity({v,f.to}) < INF && mate[v] != f.to)
                    bridges[tenacity({v,f.to})].push_back({v,f.to});
        }
        */

        for (int i = 0; i < supportTop[0]; ++i){
            auto v = support[i];
            if (v == (uint32_t)ddfsResult) continue; //skip bud
            graph.myBridge[v] = curBridge;
            graph.bud.linkTo(v,(uint32_t)ddfsResult);

            //this part of code is only needed when bottleneck found, but it doesn't mess up anything when called on two paths 
            setLvl(graph,v,2*i+1-minlvl(graph,v));
            unsigned int start = graph.srcPtr[v];
            unsigned int end = graph.srcPtr[v + 1];
            unsigned int edgeIndex;
            for(edgeIndex=start; edgeIndex < end; edgeIndex++) {
                if(graph.evenlvl[v] > graph.oddlvl[v] && 
                    graph.edgeStatus[edgeIndex] == Bridge && 
                    tenacity(graph,v,graph.dst[edgeIndex]) < INF &&
                    graph.matching[v] != graph.dst[edgeIndex]) {
                        graph.bridgeTenacity[edgeIndex] = tenacity(graph,v,graph.dst[edgeIndex]);
                }
            }
        }

        if(ddfsResult >> 32 != (uint32_t)ddfsResult) {
            augumentPath(graph,color, removedVerticesQueue, removedVerticesQueueBack, ddfsResult >> 32,(uint32_t)ddfsResult,true);
            graph.foundPath[0] = true;

            while(removedVerticesQueueBack[0]-removedVerticesQueueFront[0]) {
                
                int v = removedVerticesQueue[removedVerticesQueueFront[0]++];
                unsigned int start = graph.srcPtr[v];
                unsigned int end = graph.srcPtr[v + 1];
                unsigned int edgeIndex;
                for(edgeIndex=start; edgeIndex < end; edgeIndex++) {
                    if(graph.edgeStatus[edgeIndex] == Prop && 
                        minlvl(graph,graph.dst[edgeIndex]) > minlvl(graph,v) && 
                        tenacity(graph,v,graph.dst[edgeIndex]) < INF &&
                        !(graph.removed[graph.dst[edgeIndex]])){
                        
                        // Check last condition only if all 4 preceding are true
                        int predecessorsSize = 0;
                        unsigned int startOther = graph.srcPtr[graph.dst[edgeIndex]];
                        unsigned int endOther = graph.srcPtr[graph.dst[edgeIndex] + 1];
                        unsigned int edgeIndexOther;
                        for(edgeIndexOther=startOther; edgeIndexOther < endOther; edgeIndexOther++) {
                            if (graph.pred[edgeIndexOther])
                                predecessorsSize++;
                        }
                        if(++removedPredecessorsSize[graph.dst[edgeIndex]] == predecessorsSize) {
                            removeAndPushToQueue(graph,removedVerticesQueue,removedVerticesQueueBack,graph.dst[edgeIndex]);
                        }
                    }
                }
            }
        }
        /*
        if(ddfsResult.first != ddfsResult.second) {
            augumentPath(ddfsResult.first,ddfsResult.second,true);
            foundPath = true;
            while(!removedVerticesQueue.empty()) {
                int v = removedVerticesQueue.front();
                removedVerticesQueue.pop();
                for(auto e : graph[v])
                    if(e.type == Prop && minlvl(e.to) > minlvl(v) && !removed[e.to] && ++removedPredecessorsSize[e.to] == predecessors[e.to].size())
                        removeAndPushToQueue(e.to);
            }
        }
        */
    }
}
