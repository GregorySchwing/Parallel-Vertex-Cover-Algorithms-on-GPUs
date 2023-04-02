#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void LocalStacks_global_DFS_kernel(Stacks stacks, CSRGraph graph, unsigned int * minimum, int* global_memory, Counters* counters, 
    unsigned int * pathCounter, int* NODES_PER_SM, int startingDepth) {
#else
__global__ void LocalStacks_shared_DFS_kernel(Stacks stacks, CSRGraph graph, unsigned int * minimum, Counters* counters, unsigned int * pathCounter, 
    int* NODES_PER_SM, int startingDepth) {
#endif

    __shared__ Counters blockCounters;
    initializeCounters(&blockCounters);

    #if USE_COUNTERS
        __shared__ unsigned int sm_id;
        if (threadIdx.x==0){
            sm_id=get_smid();
        }
    #endif

    do{
        __shared__ unsigned int path;
        if(threadIdx.x==0){
            path = atomicAdd(pathCounter,1);
        }
        __syncthreads();

        // This should still work for DFS.
        // It may not though since the tree is not binary.
        if(path>=(1<<startingDepth)){
            break;
        }

        // Initialize the vertexDegrees_s
        unsigned int numDeletedVertices = 0;
        unsigned int numDeletedVertices2;

        #if USE_GLOBAL_MEMORY
        int * vertexDegrees_s = &global_memory[graph.vertexNum*(2*blockIdx.x)];
        int * vertexDegrees_s2 = &global_memory[graph.vertexNum*(2*blockIdx.x + 1)];
        #else
        extern __shared__ int shared_mem[];
        int * vertexDegrees_s = shared_mem;
        int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];
        #endif

        for(unsigned int i = threadIdx.x; i < graph.vertexNum; i += blockDim.x) {
            vertexDegrees_s[i] = graph.degree[i];
        }
        __syncthreads();

        // Find the block's sub-tree
        bool vcFound = false;
        bool minExceeded = false;
        
        // Perform first few levels to get all leaves to certain depth.
        // s.t. number of leaf nodes == numBlocks.
        // Will have to do breadth-first DFS until a sufficient number of trees exist.
        // Will do redundant work in each block to avoid inter-block communication. 
        for(unsigned int k = 0; k < startingDepth && !vcFound && !minExceeded; ++k) {
            #if USE_COUNTERS
                if (threadIdx.x==0){
                    atomicAdd(&NODES_PER_SM[sm_id],1);
                }
            #endif

            // reduction rule
            __shared__ unsigned int minimum_s;

            if(threadIdx.x == 0) {
                minimum_s = atomicOr(minimum,0);
            }

            unsigned int iterationCounter = 0, numDeletedVerticesLeaf, numDeletedVerticesTriangle, numDeletedVerticesHighDegree;
            do{
                startTime(LEAF_REDUCTION,&blockCounters);
                numDeletedVerticesLeaf = leafReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
                endTime(LEAF_REDUCTION,&blockCounters);
                numDeletedVertices += numDeletedVerticesLeaf;
                if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesHighDegree > 0){
                    startTime(TRIANGLE_REDUCTION,&blockCounters);
                    numDeletedVerticesTriangle = triangleReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
                    endTime(TRIANGLE_REDUCTION,&blockCounters);
                    numDeletedVertices += numDeletedVerticesTriangle;
                } else {
                    numDeletedVerticesTriangle = 0;
                }
                if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesTriangle > 0){
                    startTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                    numDeletedVerticesHighDegree = highDegreeReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2,numDeletedVertices,minimum_s);
                    endTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                    numDeletedVertices += numDeletedVerticesHighDegree;
                }else {
                    numDeletedVerticesHighDegree = 0;
                }
            }while(numDeletedVerticesTriangle > 0 || numDeletedVerticesHighDegree > 0);

            if(threadIdx.x == 0) {
                minimum_s = atomicOr(minimum,0);
            }
            __syncthreads();
            
            if(numDeletedVertices >= minimum_s) {
                minExceeded = true;
            } else {

                unsigned int maxVertex = 0;
                int maxDegree = 0;

                findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
                __syncthreads();

                if(maxDegree == 0){
                    vcFound = true;
                    if(threadIdx.x == 0) {
                        atomicMin(minimum, numDeletedVertices);
                    }
                    __syncthreads();
                } else {
                    unsigned int goLeft = !((path >> k) & 1u);
                    if(goLeft){
                        
                        if (threadIdx.x == 0){
                            vertexDegrees_s[maxVertex] = -1;
                        }
                        ++numDeletedVertices;
                        __syncthreads();

                        for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1]; edge+=blockDim.x) {
                            unsigned int neighbor = graph.dst[edge];
                            if(vertexDegrees_s[neighbor] != -1){
                                atomicSub(&vertexDegrees_s[neighbor], 1);
                            }
                        }

                    } else {// Delete Neighbors of maxVertex
                        numDeletedVertices += maxDegree;

                        for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1]; edge += blockDim.x) {
                            unsigned int neighbor = graph.dst[edge];
                            if(vertexDegrees_s[neighbor] != -1) {
                                for(unsigned int neighborEdge = graph.srcPtr[neighbor]; neighborEdge < graph.srcPtr[neighbor + 1]; ++neighborEdge) {
                                    unsigned int neighborOfNeighbor = graph.dst[neighborEdge];
                                    if(vertexDegrees_s[neighborOfNeighbor] != -1) {
                                        atomicSub(&vertexDegrees_s[neighborOfNeighbor], 1);
                                    }
                                }
                            }
                        }
                        __syncthreads();

                        for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1] ; edge += blockDim.x) {
                            unsigned int neighbor = graph.dst[edge];
                            vertexDegrees_s[neighbor] = -1;
                        }
                    }
                }

            }
            __syncthreads();
        }
        
        // Each block its at it's required level which is at most Root depth
        if(!vcFound && !minExceeded) {
            unsigned int stackSize = (stacks.minimum + 1);
            volatile int * stackVertexDegrees = &stacks.stacks[blockIdx.x * stackSize * graph.vertexNum];
            volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];
            int stackTop = -1;

            // go into while loop 
            bool popNextItr = false;
            __syncthreads();

            do{
                if(popNextItr){
                    startTime(POP_FROM_STACK,&blockCounters);
                    popStack(graph.vertexNum, vertexDegrees_s, &numDeletedVertices, stackVertexDegrees, stackNumDeletedVertices, &stackTop);
                    endTime(POP_FROM_STACK,&blockCounters);

                    #if USE_COUNTERS
                        if (threadIdx.x==0){
                            atomicAdd(&NODES_PER_SM[sm_id],1);
                        }
                    #endif
                }
                __syncthreads();

                //reduction rule
                __shared__ unsigned int minimum_s;

                if(threadIdx.x == 0) {
                    minimum_s = atomicOr(minimum,0);
                }
        
                unsigned int iterationCounter = 0, numDeletedVerticesLeaf, numDeletedVerticesTriangle, numDeletedVerticesHighDegree;
                do{
                    startTime(LEAF_REDUCTION,&blockCounters);
                    numDeletedVerticesLeaf = leafReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
                    endTime(LEAF_REDUCTION,&blockCounters);
                    numDeletedVertices += numDeletedVerticesLeaf;
                    if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesHighDegree > 0){
                        startTime(TRIANGLE_REDUCTION,&blockCounters);
                        numDeletedVerticesTriangle = triangleReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
                        endTime(TRIANGLE_REDUCTION,&blockCounters);
                        numDeletedVertices += numDeletedVerticesTriangle;
                    } else {
                        numDeletedVerticesTriangle = 0;
                    }
                    if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesTriangle > 0){
                        startTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                        numDeletedVerticesHighDegree = highDegreeReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2,numDeletedVertices,minimum_s);
                        endTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                        numDeletedVertices += numDeletedVerticesHighDegree;
                    }else {
                        numDeletedVerticesHighDegree = 0;
                    }
                }while(numDeletedVerticesTriangle > 0 || numDeletedVerticesHighDegree > 0);
        
                if(threadIdx.x == 0) {
                    minimum_s = *minimum;
                }
                __syncthreads();

                if(numDeletedVertices >= minimum_s){
                    popNextItr = true;
                } else {

                    // Find max degree
                    unsigned int maxVertex = 0;
                    int maxDegree = 0;
                    startTime(MAX_DEGREE,&blockCounters);
                    findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
                    endTime(MAX_DEGREE,&blockCounters);
                    __syncthreads();

                    if(maxDegree == 0){
                        if(threadIdx.x == 0) {
                            atomicMin(minimum, numDeletedVertices);
                        }
                        __syncthreads();
                        popNextItr = true;
                    } else {

                        popNextItr = false;
                    
                        __syncthreads();
                        startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                        deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                        endTime(PREPARE_RIGHT_CHILD,&blockCounters);
                        __syncthreads();
                        startTime(PUSH_TO_STACK,&blockCounters);
                        pushStack(graph.vertexNum, vertexDegrees_s2, &numDeletedVertices2, stackVertexDegrees, stackNumDeletedVertices, &stackTop);
                        endTime(PUSH_TO_STACK,&blockCounters);
                        __syncthreads();
                        startTime(PREPARE_LEFT_CHILD,&blockCounters);
                        deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
                        endTime(PREPARE_LEFT_CHILD,&blockCounters);
                    }
                }
                __syncthreads();

            } while(stackTop != -1);
        }
    }while(true);

    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif
}
