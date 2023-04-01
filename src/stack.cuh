#ifndef STACK_H
#define STACK_H

struct Stacks{
    volatile int * stacks;
    volatile unsigned int * stacksNumDeletedVertices;
    volatile unsigned int * backtrackingIndices;
    int minimum;
};

__device__ void popStack(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices, volatile int * stackVertexDegrees, 
    volatile unsigned int* stackNumDeletedVertices, int * stackTop){

    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        vertexDegrees_s[vertex] = stackVertexDegrees[(*stackTop)*vertexNum + vertex];
    }

    *numDeletedVertices = stackNumDeletedVertices[*stackTop];
    
    --(*stackTop);
}

__device__ void pushStack(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices, volatile int * stackVertexDegrees, 
    volatile unsigned int* stackNumDeletedVertices, int * stackTop){

    ++(*stackTop);
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum ; vertex += blockDim.x) {
        stackVertexDegrees[(*stackTop)*vertexNum + vertex] = vertexDegrees_s[vertex];
    }
    if(threadIdx.x == 0) {
        stackNumDeletedVertices[*stackTop] = *numDeletedVertices;
    }
}

Stacks allocateStacks(int vertexNum, int numBlocks, unsigned int minimum){
    Stacks stacks;

    volatile int* stacks_d;
    volatile unsigned int* stacksNumDeletedVertices_d;
    volatile unsigned int* backtrackingIndices_d;
    /*
        Not sure what the rationale is here.
        Seems like every block could solve the entire tree independently
        in it's respective memory, since the most vertices possible added
        to the cover is v0 and each block as V*v0 memory.
        Block 0
            array 0; [0 1 2 ... V]
            array 1; [0 1 2 ... V]
            array 2; [0 1 2 ... V]
            ...
            array v0; [0 1 2 ... V] 
            value 0; 0
            value 1; 0
            value 2; 0
            ...
            value v0; 0

        Block 1
            array 0; [0 1 2 ... V]
            array 1; [0 1 2 ... V]
            array 2; [0 1 2 ... V]
            ...
            array v0; [0 1 2 ... V] 
            value 0; 0
            value 1; 0
            value 2; 0
            ...
            value v0; 0
    
    */

    // Each active block maintains (minimum+1) copies of the data array [V]
    cudaMalloc((void**) &stacks_d, (minimum + 1) * (vertexNum) * sizeof(int) * numBlocks);
    // Each active block maintains (minimum+1) values associated with each data array.
    cudaMalloc((void**) &stacksNumDeletedVertices_d, (minimum + 1) * sizeof(unsigned int) * numBlocks);
    cudaMalloc((void**) &backtrackingIndices_d, vertexNum * sizeof(unsigned int) * numBlocks);

    stacks.stacks = stacks_d;
    stacks.stacksNumDeletedVertices = stacksNumDeletedVertices_d;
    stacks.backtrackingIndices = backtrackingIndices_d;
    stacks.minimum = minimum;

    return stacks;
}

void cudaFreeStacks(Stacks stacks){
    cudaFree((void*)stacks.stacks);
    cudaFree((void*)stacks.stacksNumDeletedVertices);
    cudaFree((void*)stacks.backtrackingIndices);
}

#endif