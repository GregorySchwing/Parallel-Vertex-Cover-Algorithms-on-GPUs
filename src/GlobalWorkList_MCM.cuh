
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_kernel(Stacks stacks, unsigned int * maximum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_kernel(Stacks stacks, unsigned int * maximum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* NODES_PER_SM) {
#endif

    __shared__ Counters blockCounters;
    initializeCounters(&blockCounters);

    #if USE_COUNTERS
        __shared__ unsigned int sm_id;
        if (threadIdx.x==0){
            sm_id=get_smid();
        }
    #endif

    int stackTop = -1;
    unsigned int stackSize = (stacks.minimum + 1);
    // This is still degrees.
    volatile int * stackVertexDegrees = &stacks.stacks[blockIdx.x * stackSize * graph.vertexNum];
    // Starts at zero.  I use this as an edge index of the currently considered edge.
    volatile unsigned int * stackCurrentVertexIndex = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];

    // Define the vertexDegree_s
    unsigned int currentVertexIndex;
    unsigned int currentVertexIndex2;
    
    #if USE_GLOBAL_MEMORY
    int * vertexDegrees_s = &global_memory[graph.vertexNum*(2*blockIdx.x)];
    int * vertexDegrees_s2 = &global_memory[graph.vertexNum*(2*blockIdx.x + 1)];
    #else
    extern __shared__ int shared_mem[];
    int * vertexDegrees_s = shared_mem;
    int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];
    #endif

    bool dequeueOrPopNextItr = true; 
    __syncthreads();

    __shared__ bool first_to_dequeue;
    if (threadIdx.x==0){
        if(atomicCAS(first_to_dequeue_global,0,1) == 0) { 
            first_to_dequeue = true;
        } else {
            first_to_dequeue = false;
        }
    }
    __syncthreads();
    if (first_to_dequeue){
        for(unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex += blockDim.x) {
            vertexDegrees_s[vertex]=workList.list[vertex];
        }
        currentVertexIndex = workList.listNumDeletedVertices[0];
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    
    while(true){
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack(graph.vertexNum, vertexDegrees_s, &currentVertexIndex, stackVertexDegrees, stackCurrentVertexIndex, &stackTop);               
                endTime(POP_FROM_STACK,&blockCounters);

                #if USE_COUNTERS
                    if (threadIdx.x==0){
                        atomicAdd(&NODES_PER_SM[sm_id],1);
                    }
                #endif
                __syncthreads();
            } else { // Local stack is empty, read from the global workList
                startTime(TERMINATE,&blockCounters);
                startTime(DEQUEUE,&blockCounters);
                if(!dequeue(vertexDegrees_s, workList, graph.vertexNum, &currentVertexIndex)) {   
                    endTime(TERMINATE,&blockCounters);
                    break;
                }
                endTime(DEQUEUE,&blockCounters);

                #if USE_COUNTERS
                    if (threadIdx.x==0){
                        atomicAdd(&NODES_PER_SM[sm_id],1);
                    }
                #endif
            }
        }

        __shared__ unsigned int maximum_s;
        if(threadIdx.x == 0) {
            maximum_s = atomicOr(maximum,0);
        }

        __syncthreads();

        // I use this to find the size of the matching based on the degrees.
        // Recall I used my one scalar to track edge index considered.
        unsigned int numMatchedVertices = getMatchedVertices(graph.vertexNum, vertexDegrees_s, vertexDegrees_s2);
        startTime(MAX_DEGREE,&blockCounters);
        //findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
        // Find the source containing the current edge considered
        // Find the next edge between two vertices with non-zero degree.
        //unsigned int nextEdge = findNextEdge(graph.srcPtrUncompressed, graph.dst, vertexDegrees_s, currentEdgeIndex, 2*graph.edgeNum, graph.vertexNum);            
        //unsigned int nextEdge = findNextEdgeByWarp(graph.srcPtrUncompressed, graph.dst, vertexDegrees_s, currentEdgeIndex, 2*graph.edgeNum, graph.vertexNum);    
        unsigned int nextVertex = findNextVertexByWarp(graph.srcPtrUncompressed, graph.dst, vertexDegrees_s, currentVertexIndex, graph.vertexNum);    
                
        /*
        if (threadIdx.x==0)
            printf("%d %d \n",nextEdge,nextEdgeFast);
        assert(nextEdge==nextEdgeFast);
        */
        endTime(MAX_DEGREE,&blockCounters);
        if(threadIdx.x == 0) {
            maximum_s = atomicOr(maximum,0);
            //atomicMax(maximum, numMatchedVertices);
            //printf("block id %d currentEdgeIndex %d nextEdge %d maximum_s %d graph.vertexNum %d\n",blockIdx.x,currentEdgeIndex, nextEdge,maximum_s,graph.vertexNum);
        }
        __syncthreads();
        // I found a perfect matching
        if(numMatchedVertices == graph.vertexNum 
        // Someone else found a perfect matching
        || maximum_s == graph.vertexNum 
        // Allowable on when matching is empty. Otherwise, indicates node lacks children
        || nextVertex == graph.vertexNum) { // Reached the bottom of the tree, maximum vertex cover possibly found
            if(threadIdx.x==0){
                //atomicMin(maximum, currentEdgeIndex);
                //printf("atomicMax block id %d numMatchedVertices %u maximum_s %d, current edge %u graph.vertexNum/2 %d\n",blockIdx.x,numMatchedVertices, maximum_s, currentEdgeIndex,graph.vertexNum/2);
                atomicMax(maximum, numMatchedVertices);
            }
            //if (threadIdx.x==0)
            //printf("popping 2 block id %d upperBoundAddableEdges %u numMatchedVertices %u maximum_s %d, current edge %u graph.vertexNum/2 %d\n",blockIdx.x,upperBoundAddableEdges,numMatchedVertices, maximum_s, currentEdgeIndex,graph.vertexNum/2);

            dequeueOrPopNextItr = true;

        } else { // Vertex cover not found, need to branch

            startTime(PREPARE_RIGHT_CHILD,&blockCounters);
            //deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &currentEdgeIndex, vertexDegrees_s2, &currentEdgeIndex2, maxDegree, maxVertex);
            skipEdge(graph,vertexDegrees_s, &currentVertexIndex, vertexDegrees_s2, &currentVertexIndex2, nextVertex);
            endTime(PREPARE_RIGHT_CHILD,&blockCounters);
            //if (threadIdx.x==0)
            //    printf("pushing block id %d currentEdgeIndex2 %d\n",blockIdx.x,currentEdgeIndex2);

            __syncthreads();

            bool enqueueSuccess;
            if(checkThreshold(workList)){
                startTime(ENQUEUE,&blockCounters);
                enqueueSuccess = enqueue(vertexDegrees_s2, workList, graph.vertexNum, &currentVertexIndex2);
            } else  {
                enqueueSuccess = false;
            }
            
            __syncthreads();
            
            if(!enqueueSuccess) {
                startTime(PUSH_TO_STACK,&blockCounters);
                pushStack(graph.vertexNum, vertexDegrees_s2, &currentVertexIndex2, stackVertexDegrees, stackCurrentVertexIndex, &stackTop);                    
                maxDepth(stackTop, &blockCounters);
                endTime(PUSH_TO_STACK,&blockCounters);
                __syncthreads(); 
            } else {
                endTime(ENQUEUE,&blockCounters);
            }

            startTime(PREPARE_LEFT_CHILD,&blockCounters);
            // Prepare the child that removes the neighbors of the max vertex to be processed on the next iteration
            deleteNextEdge(graph, vertexDegrees_s, currentVertexIndex);
            endTime(PREPARE_LEFT_CHILD,&blockCounters);
            dequeueOrPopNextItr = false;
        }
        __syncthreads();
    }

    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif
}
