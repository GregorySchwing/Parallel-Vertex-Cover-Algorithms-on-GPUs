
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
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
    volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];

    // Define the vertexDegree_s
    unsigned int numDeletedVertices;
    unsigned int numDeletedVertices2;
    
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
        numDeletedVertices = workList.listNumDeletedVertices[0];
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    
    while(true){
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack(graph.vertexNum, vertexDegrees_s, &numDeletedVertices, stackVertexDegrees, stackNumDeletedVertices, &stackTop);               
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
                if(!dequeue(vertexDegrees_s, workList, graph.vertexNum, &numDeletedVertices)) {   
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
            minimum_s = atomicOr(minimum,0);
        }

        __syncthreads();

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
            ++iterationCounter;
        }while(numDeletedVerticesTriangle > 0 || numDeletedVerticesHighDegree > 0);
        
        unsigned int numOfEdges = findNumOfEdges(graph.vertexNum, vertexDegrees_s, vertexDegrees_s2);

        if(threadIdx.x == 0) {
            minimum_s = atomicOr(minimum,0);
        }
        __syncthreads();

        if(numDeletedVertices >= minimum_s || numOfEdges>=square(minimum_s-numDeletedVertices-1)+1) { // Reached the bottom of the tree, no minimum vertex cover found
            dequeueOrPopNextItr = true;

        } else {
            unsigned int maxVertex;
            int maxDegree;
            startTime(MAX_DEGREE,&blockCounters);
            findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
            endTime(MAX_DEGREE,&blockCounters);

            __syncthreads();
            if(maxDegree == 0) { // Reached the bottom of the tree, minimum vertex cover possibly found
                if(threadIdx.x==0){
                    atomicMin(minimum, numDeletedVertices);
                }

                dequeueOrPopNextItr = true;

            } else { // Vertex cover not found, need to branch

                startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                endTime(PREPARE_RIGHT_CHILD,&blockCounters);

                __syncthreads();

                bool enqueueSuccess;
                if(checkThreshold(workList)){
                    startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue(vertexDegrees_s2, workList, graph.vertexNum, &numDeletedVertices2);
                } else  {
                    enqueueSuccess = false;
                }
                
                __syncthreads();
                
                if(!enqueueSuccess) {
                    startTime(PUSH_TO_STACK,&blockCounters);
                    pushStack(graph.vertexNum, vertexDegrees_s2, &numDeletedVertices2, stackVertexDegrees, stackNumDeletedVertices, &stackTop);                    
                    maxDepth(stackTop, &blockCounters);
                    endTime(PUSH_TO_STACK,&blockCounters);
                    __syncthreads(); 
                } else {
                    endTime(ENQUEUE,&blockCounters);
                }

                startTime(PREPARE_LEFT_CHILD,&blockCounters);
                // Prepare the child that removes the neighbors of the max vertex to be processed on the next iteration
                deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
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

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_DFS_2_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_DFS_2_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
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
    volatile int * stackBacktrackingIndices = &stacks.backtrackingIndices[blockIdx.x * stackSize * graph.vertexNum];
    volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];
    volatile unsigned int * stackDepth = &stacks.depth[blockIdx.x * stackSize];

    // Define the vertexDegree_s
    unsigned int numDeletedVertices;
    unsigned int numDeletedVertices2;
    unsigned int depth;
    unsigned int depth2;
    int * backtrackingIndices;
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
        numDeletedVertices = workList.listNumDeletedVertices[0];
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    
    while(true){
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack_DFS(graph.vertexNum, vertexDegrees_s, &numDeletedVertices, &depth, backtrackingIndices, stackVertexDegrees, stackNumDeletedVertices, stackDepth, stackBacktrackingIndices, &stackTop);               
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
                if(!dequeue(vertexDegrees_s, workList, graph.vertexNum, &numDeletedVertices)) {   
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
            minimum_s = atomicOr(minimum,0);
        }

        __syncthreads();
        
        unsigned int numOfEdges = findNumOfEdges(graph.vertexNum, vertexDegrees_s, vertexDegrees_s2);

        if(threadIdx.x == 0) {
            minimum_s = atomicOr(minimum,0);
        }
        __syncthreads();

        if(numDeletedVertices >= minimum_s || numOfEdges>=square(minimum_s-numDeletedVertices-1)+1) { // Reached the bottom of the tree, no minimum vertex cover found
            dequeueOrPopNextItr = true;

        } else {
            unsigned int maxVertex;
            int maxDegree;
            startTime(MAX_DEGREE,&blockCounters);
            findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
            endTime(MAX_DEGREE,&blockCounters);

            __syncthreads();
            if(maxDegree == 0) { // Reached the bottom of the tree, minimum vertex cover possibly found
                if(threadIdx.x==0){
                    atomicMin(minimum, numDeletedVertices);
                }

                dequeueOrPopNextItr = true;

            } else { // Vertex cover not found, need to branch

                startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                endTime(PREPARE_RIGHT_CHILD,&blockCounters);

                __syncthreads();

                bool enqueueSuccess;
                if(checkThreshold(workList)){
                    startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue(vertexDegrees_s2, workList, graph.vertexNum, &numDeletedVertices2);
                } else  {
                    enqueueSuccess = false;
                }
                
                __syncthreads();
                
                if(!enqueueSuccess) {
                    startTime(PUSH_TO_STACK,&blockCounters);
                    pushStack(graph.vertexNum, vertexDegrees_s2, &numDeletedVertices2, stackVertexDegrees, stackNumDeletedVertices, &stackTop);                    
                    maxDepth(stackTop, &blockCounters);
                    endTime(PUSH_TO_STACK,&blockCounters);
                    __syncthreads(); 
                } else {
                    endTime(ENQUEUE,&blockCounters);
                }

                startTime(PREPARE_LEFT_CHILD,&blockCounters);
                // Prepare the child that removes the neighbors of the max vertex to be processed on the next iteration
                deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
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
/*
#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_DFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_DFS_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
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
    volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];
    volatile unsigned int * stackBacktrackingIndices = &stacks.backtrackingIndices[blockIdx.x * stackSize * graph.vertexNum];
    volatile unsigned int * stackDepth = &stacks.depth[blockIdx.x * stackSize];

    // Define the vertexDegree_s
    //unsigned int numDeletedVertices;
    //unsigned int numDeletedVertices2;
    __shared__ unsigned int depth;
    __shared__ unsigned int depth2;
    __shared__ unsigned int startingVertex;
    __shared__ unsigned int startingVertex2;
    __shared__ bool foundNeighbor;
    //unsigned int * backtrackingIndices;
    #if USE_GLOBAL_MEMORY
    int * visited_s = &global_memory[graph.vertexNum*(2*blockIdx.x)];
    unsigned int * backtrackingIndices_s = (unsigned int *)&global_memory[graph.vertexNum*(2*blockIdx.x + 1)];
    #else
    extern __shared__ int shared_mem[];
    int * visited_s = shared_mem;
    unsigned int * backtrackingIndices_s = (unsigned int *)&shared_mem[graph.vertexNum];
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
        if (threadIdx.x==0)
            printf("Block ID %d first to dequue\n",blockIdx.x);
        for(unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex += blockDim.x) {
            visited_s[vertex]=workList.list[vertex];
            backtrackingIndices_s[vertex]=workList.backtrackingIndices[vertex];
        }
        startingVertex = workList.listNumDeletedVertices[0];
        depth = 0;
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    if (threadIdx.x==0)
    printf("BlockID %d\n", blockIdx.x);
    while(true){
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack_DFS(graph.vertexNum, visited_s, &startingVertex, &depth, backtrackingIndices_s, stackVertexDegrees, stackNumDeletedVertices, stackDepth, stackBacktrackingIndices, &stackTop);               
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
                
                if(!dequeue_DFS(visited_s, backtrackingIndices_s, &depth, workList, graph.vertexNum, &startingVertex)) {   
                                if(threadIdx.x == 0) 
                    printf("Block %d breaking after dequeue_DFS pop %d stackTop %d depth %d startingVertex %d bti %d \n",blockIdx.x, dequeueOrPopNextItr, stackTop, depth, startingVertex,backtrackingIndices_s[depth]);
                    endTime(TERMINATE,&blockCounters);
                    break;
                }
                            if(threadIdx.x == 0) 
                printf("Block %d continuing after dequeue_DFS pop %d stackTop %d depth %d startingVertex %d bti %d \n",blockIdx.x, dequeueOrPopNextItr, stackTop, depth, startingVertex,backtrackingIndices_s[depth]);
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
            minimum_s = atomicOr(minimum,graph.matching[startingVertex]==-1&&depth>0);
        }
        // END VC PRE-WORK.
        __syncthreads();
        if (minimum_s){
            if(threadIdx.x == 0) 
                printf("BLOCK %d RETURNING\n", blockIdx.x);
            //return;
        } else {
            //printf("BLOCK %d NOT RETURNING\n", blockIdx.x);
        }
        if(threadIdx.x == 0) 
            printf("Block %d waiting after return depth %d startingVertex %d bti %d \n",blockIdx.x, depth, startingVertex,backtrackingIndices_s[depth]);

        // Reached the bottom of the tree, no minimum vertex cover found
        if(minimum_s || backtrackingIndices_s[depth]>=graph.srcPtr[startingVertex+1]-graph.srcPtr[startingVertex]) {
            dequeueOrPopNextItr = true;
            if (threadIdx.x==0)
            printf("Block %d No Neighbors left of vertex %d start %d end %d bti %d depth %d\n", blockIdx.x, startingVertex, graph.srcPtr[startingVertex],graph.srcPtr[startingVertex+1], backtrackingIndices_s[depth], depth);
        } else {
            startTime(MAX_DEGREE,&blockCounters);
            if (threadIdx.x==0)
            printf("Block %d Neighbors left of vertex %d start %d end %d bti %d depth %d\n", blockIdx.x, startingVertex, graph.srcPtr[startingVertex],graph.srcPtr[startingVertex+1], backtrackingIndices_s[depth], depth);
            findNextVertex(graph,visited_s, &startingVertex, backtrackingIndices_s, &depth, &startingVertex2, &depth2,&foundNeighbor);
            endTime(MAX_DEGREE,&blockCounters);

            __syncthreads();
            // Reached the bottom of the tree
            // LOGIC -> dequeueOrPopNextItr = true

            // Found unmatched vertex
            // LOGIC -> set all blocks to terminate.

            if(!foundNeighbor) {
                if (threadIdx.x==0)
                    printf("didnt find neighbor, set pop\n");
                dequeueOrPopNextItr = true;

            } else { // Vertex cover not found, need to branch
                if (threadIdx.x==0)
                    printf("found neighbor, branch\n");

                startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                // increments current depth's backtrackingIndex, and pushes current visited_s array.
                prepare_right_child_DFS(&startingVertex, backtrackingIndices_s, &depth);
                endTime(PREPARE_RIGHT_CHILD,&blockCounters);

                __syncthreads();
                if (threadIdx.x==0)
                printf("After PRC Block %d Neighbors left of vertex %d start %d end %d bti %d depth %d\n", blockIdx.x, startingVertex, graph.srcPtr[startingVertex],graph.srcPtr[startingVertex+1], backtrackingIndices_s[depth], depth);
                bool enqueueSuccess;
                if(checkThreshold(workList)){
                    startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue_DFS(visited_s, backtrackingIndices_s,&depth,workList, graph.vertexNum, &startingVertex);
                } else  {
                    enqueueSuccess = false;
                }
                
                __syncthreads();
                
                if(!enqueueSuccess) {
                    startTime(PUSH_TO_STACK,&blockCounters);
                    pushStack_DFS(graph.vertexNum, visited_s, &startingVertex, &depth, backtrackingIndices_s, stackVertexDegrees, stackNumDeletedVertices, stackDepth, stackBacktrackingIndices, &stackTop);                    
                    maxDepth(stackTop, &blockCounters);
                    endTime(PUSH_TO_STACK,&blockCounters);
                    __syncthreads(); 
                } else {
                    endTime(ENQUEUE,&blockCounters);
                }

                startTime(PREPARE_LEFT_CHILD,&blockCounters);
                // Decrements the current depth's bti and sets the next vertex as visited.
                prepare_left_child_DFS(visited_s, backtrackingIndices_s, &startingVertex, &depth, &startingVertex2,&depth2);
                __syncthreads(); 
                if (threadIdx.x==0)
                printf("After PLC Block %d Neighbors left of vertex %d start %d end %d bti %d depth %d\n", blockIdx.x, startingVertex, graph.srcPtr[startingVertex],graph.srcPtr[startingVertex+1], backtrackingIndices_s[depth], depth);
                //deleteMaxDegreeVertex(graph, visited_s, &numDeletedVertices, maxVertex);
                // Replace this with setting the visited, incrementing the bti's, and incrementing depth.
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
*/