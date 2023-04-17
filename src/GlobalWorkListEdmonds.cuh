
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.h"

#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkList_global_Edmonds_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
    int* first_to_dequeue_global, int* global_memory, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkList_shared_Edmonds_kernel(Stacks stacks, unsigned int * minimum, WorkList workList, CSRGraph graph, Counters* counters, 
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
        for(unsigned int vertex = threadIdx.x; vertex < graph.num_unmatched_vertices[0]; vertex += blockDim.x) {
            vertexDegrees_s[vertex]=graph.unmatched_vertices[vertex];
        }
        numDeletedVertices = graph.num_unmatched_vertices[0];
        dequeueOrPopNextItr = true;

        __syncthreads();

        bool enqueueSuccess;
        for(unsigned int vertex = 0; vertex < graph.num_unmatched_vertices[0]; vertex += 1) {

            if(checkThreshold(workList)){
                startTime(ENQUEUE,&blockCounters);
                enqueueSuccess = enqueue(&vertexDegrees_s[vertex], workList, 1, &vertex);
            } else  {
                enqueueSuccess = false;
            }
        
            __syncthreads();
            if (threadIdx.x==0)
                printf("Pushed %d\n",vertexDegrees_s[vertex]);
        }
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
                if(!dequeue(vertexDegrees_s, workList, 1, &numDeletedVertices)) {   
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
            
            if (threadIdx.x==0)
                printf("Popped %d\n",vertexDegrees_s[0]);
        }


        __shared__ unsigned int minimum_s;
        if(threadIdx.x == 0) {
            minimum_s = atomicOr(minimum,0);
        }

        __syncthreads();

        // Do some work here.
        if(!dequeue(vertexDegrees_s, workList, 1, &numDeletedVertices)) {   
            endTime(TERMINATE,&blockCounters);
            break;
        }
        return;

        __syncthreads();
    }

    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif
}
