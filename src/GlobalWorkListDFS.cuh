
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_DFS_kernel(Stacks stacks, unsigned int * minimum, unsigned int * solution_mutex, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_DFS_kernel(Stacks stacks, unsigned int * minimum, unsigned int * solution_mutex, WorkList workList, CSRGraph graph, Counters* counters, 
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
    volatile int * stackVertexDegrees = &stacks.stacks[blockIdx.x * stackSize * graph.vertexNum];
    volatile uint64_t* stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];

    // stored in the lower 32 bits of the stackNumDel
    unsigned int numDeletedVertices;
    unsigned int numDeletedVertices2;
    // stored in the upper 32 bits of the stackNumDel
    unsigned int edgeIndex;
    unsigned int edgeIndex2;

    // Define the vertexDegree_s

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
        numDeletedVertices = (uint32_t)workList.listNumDeletedVertices[0];
        edgeIndex = (workList.listNumDeletedVertices[0] >> 32);
        vertexDegrees_s[numDeletedVertices]=1;
        //if (threadIdx.x == 0)
        //    printf("numDeletedVertices %d edgeIndex %d \n", numDeletedVertices, edgeIndex);
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    
    while(true){
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack_DFS(graph.vertexNum, vertexDegrees_s, &numDeletedVertices, &edgeIndex, stackVertexDegrees, stackNumDeletedVertices, &stackTop);               
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
                if(!dequeue_DFS(vertexDegrees_s, workList, graph.vertexNum, &numDeletedVertices, &edgeIndex)) {   
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


        __shared__ unsigned int minimum_s;
        if(threadIdx.x == 0) {
            minimum_s = atomicOr(minimum,graph.matching[numDeletedVertices]==-1&&!vertexDegrees_s[numDeletedVertices]);
        }

        __syncthreads();
        
        //unsigned int numOfEdges = findNumOfEdges(graph.vertexNum, vertexDegrees_s, vertexDegrees_s2);

        if(threadIdx.x == 0) {
            minimum_s = atomicOr(minimum,graph.matching[numDeletedVertices]==-1&&!vertexDegrees_s[numDeletedVertices]);
        }
        
        if(threadIdx.x == 0) {
            if(graph.matching[numDeletedVertices]==-1&&!vertexDegrees_s[numDeletedVertices]){
                if(0 == atomicCAS(solution_mutex, 0, 1)){
                    // Write stack of starting vertices to solution array.
                    //copySolution(graph.vertexNum,graph.solution,graph.solution_length,graph.solution_last_vertex,&numDeletedVertices,vertexDegrees_s,&depth);
                    printf("Found solution of starting with %d and ending with %d\n",graph.unmatched_vertices[0],numDeletedVertices);
                }
            }
        }
        
        __syncthreads();

        // Either another block found a path or 
        // this block reached a dead-end
        if(minimum_s||edgeIndex>=(graph.srcPtr[numDeletedVertices+1]-graph.srcPtr[numDeletedVertices])) { // Reached the bottom of the tree, no minimum vertex cover found
            dequeueOrPopNextItr = true;
        } else {
            unsigned int maxVertex;
            int maxDegree;
            __shared__ unsigned int neighbor;
            __shared__ unsigned int foundNeighbor;
            foundNeighbor = 0;
            startTime(MAX_DEGREE,&blockCounters);
            //findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
            findUnmatchedNeighbor(graph,numDeletedVertices,&edgeIndex,&neighbor,&foundNeighbor,vertexDegrees_s);
            endTime(MAX_DEGREE,&blockCounters);

            __syncthreads();
            if(!foundNeighbor) { // Reached the bottom of the tree, minimum vertex cover possibly found
                /*
                if(threadIdx.x==0){
                    atomicMin(minimum, numDeletedVertices);
                }
                */
                dequeueOrPopNextItr = true;

            } else { // Vertex cover not found, need to branch


                startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                // Right child increments backtracking indices without incrementing depth
                // this doesnt set any visited flags
                //deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                //deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                prepareRightChild(graph,vertexDegrees_s,&numDeletedVertices,&edgeIndex,vertexDegrees_s2,&numDeletedVertices2,&edgeIndex2);
                endTime(PREPARE_RIGHT_CHILD,&blockCounters);


                __syncthreads();

                bool enqueueSuccess;
                if(checkThreshold(workList)){
                    startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue_DFS(vertexDegrees_s2, workList, graph.vertexNum, &numDeletedVertices2,&edgeIndex2);
                } else  {
                    enqueueSuccess = false;
                }
                
                __syncthreads();
                
                if(!enqueueSuccess) {
                    startTime(PUSH_TO_STACK,&blockCounters);
                    pushStack_DFS(graph.vertexNum, vertexDegrees_s2, &numDeletedVertices2, &edgeIndex2,stackVertexDegrees, stackNumDeletedVertices, &stackTop);                    
                    maxDepth(stackTop, &blockCounters);
                    endTime(PUSH_TO_STACK,&blockCounters);
                    __syncthreads(); 
                } else {
                    endTime(ENQUEUE,&blockCounters);
                }

                startTime(PREPARE_LEFT_CHILD,&blockCounters);
                // Prepare the child that removes the neighbors of the max vertex to be processed on the next iteration
                //deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
                // Left child increments leaves backtracking indices as is
                // sets foundNeighbor flag, increments depth, and sets startVertex to neighbor
                //deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
                prepareLeftChild(vertexDegrees_s,&numDeletedVertices,&edgeIndex,&neighbor);
                endTime(PREPARE_LEFT_CHILD,&blockCounters);

                dequeueOrPopNextItr = false;
            }
        }
        __syncthreads();
    }

    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif
}
