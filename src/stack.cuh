#ifndef STACK_H
#define STACK_H

struct Stacks{
    volatile int * stacks;
    volatile uint32_t* stacks_bits;
    volatile uint64_t* stacksNumDeletedVertices;
    int minimum;
};

__device__ void popStack(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices, volatile int * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        vertexDegrees_s[vertex] = stackVertexDegrees[(*stackTop)*vertexNum + vertex];
    }

    *numDeletedVertices = stackNumDeletedVertices[*stackTop];
    
    --(*stackTop);
}

__device__ void popStack_DFS(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int* edgeIndex,volatile int * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        vertexDegrees_s[vertex] = stackVertexDegrees[(*stackTop)*vertexNum + vertex];
    }

    *numDeletedVertices = (uint32_t)stackNumDeletedVertices[*stackTop];
    *edgeIndex = (stackNumDeletedVertices[*stackTop] >> 32);

    --(*stackTop);
}


__device__ void popStack_DFS_bits(unsigned int vertexNum, uint32_t* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int* edgeIndex,volatile uint32_t * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        vertexDegrees_s[vertex] = stackVertexDegrees[(*stackTop)*vertexNum + vertex];
    }

    *numDeletedVertices = (uint32_t)stackNumDeletedVertices[*stackTop];
    *edgeIndex = (stackNumDeletedVertices[*stackTop] >> 32);

    --(*stackTop);
}



__device__ void pushStack(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices, volatile int * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    ++(*stackTop);
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum ; vertex += blockDim.x) {
        stackVertexDegrees[(*stackTop)*vertexNum + vertex] = vertexDegrees_s[vertex];
    }
    if(threadIdx.x == 0) {
        stackNumDeletedVertices[*stackTop] = *numDeletedVertices;
    }
}


__device__ void pushStack_DFS(unsigned int vertexNum, int* vertexDegrees_s, unsigned int* numDeletedVertices,unsigned int* edgeIndex, volatile int * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    ++(*stackTop);
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum ; vertex += blockDim.x) {
        stackVertexDegrees[(*stackTop)*vertexNum + vertex] = vertexDegrees_s[vertex];
    }
    if(threadIdx.x == 0) {
        uint32_t leastSignificantWord = *numDeletedVertices;
        uint32_t mostSignificantWord = *edgeIndex;
        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
        stackNumDeletedVertices[*stackTop] = edgePair;
    }
}


__device__ void pushStack_DFS_bits(unsigned int vertexNum, uint32_t* vertexDegrees_s, unsigned int* numDeletedVertices,unsigned int* edgeIndex, volatile uint32_t * stackVertexDegrees, 
    volatile uint64_t* stackNumDeletedVertices, int * stackTop){

    ++(*stackTop);
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum ; vertex += blockDim.x) {
        stackVertexDegrees[(*stackTop)*vertexNum + vertex] = vertexDegrees_s[vertex];
    }
    if(threadIdx.x == 0) {
        uint32_t leastSignificantWord = *numDeletedVertices;
        uint32_t mostSignificantWord = *edgeIndex;
        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
        stackNumDeletedVertices[*stackTop] = edgePair;
    }
}


Stacks allocateStacks(int vertexNum, int numBlocks, unsigned int minimum){
    Stacks stacks;

    volatile int* stacks_d;
    volatile uint64_t* stacksNumDeletedVertices_d;
    cudaMalloc((void**) &stacks_d, (minimum + 1) * (vertexNum) * sizeof(int) * numBlocks);
    cudaMalloc((void**) &stacksNumDeletedVertices_d, (minimum + 1) * sizeof(unsigned int) * numBlocks);

    stacks.stacks = stacks_d;
    stacks.stacksNumDeletedVertices = stacksNumDeletedVertices_d;
    stacks.minimum = minimum;

    return stacks;
}



Stacks allocateStacks_DFS(int vertexNum, int numBlocks, unsigned int minimum){
    Stacks stacks;

    volatile int* stacks_d;
    volatile uint64_t* stacksNumDeletedVertices_d;
    cudaMalloc((void**) &stacks_d, (minimum + 2) * (vertexNum) * sizeof(int) * numBlocks);
    cudaMalloc((void**) &stacksNumDeletedVertices_d, (minimum + 2) * sizeof(unsigned int) * numBlocks);

    stacks.stacks = stacks_d;
    stacks.stacksNumDeletedVertices = stacksNumDeletedVertices_d;
    stacks.minimum = minimum;

    return stacks;
}

Stacks allocateStacks_DFS_bits(int vertexNum, int numBlocks, unsigned int minimum){
    Stacks stacks;

    volatile uint32_t* stacks_bits_d;
    volatile uint64_t* stacksNumDeletedVertices_d;
    cudaMalloc((void**) &stacks_bits_d, (minimum + 2) * ((vertexNum+31)/32) * sizeof(uint32_t) * numBlocks);
    cudaMalloc((void**) &stacksNumDeletedVertices_d, (minimum + 1) * sizeof(unsigned int) * numBlocks);

    stacks.stacks_bits = stacks_bits_d;
    stacks.stacksNumDeletedVertices = stacksNumDeletedVertices_d;
    stacks.minimum = minimum;

    return stacks;
}

void cudaFreeStacks(Stacks stacks){
    cudaFree((void*)stacks.stacks);
    cudaFree((void*)stacks.stacksNumDeletedVertices);
}

#endif