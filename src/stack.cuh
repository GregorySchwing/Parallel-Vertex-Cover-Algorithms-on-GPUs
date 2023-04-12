#ifndef STACK_H
#define STACK_H

struct Stacks{
    volatile int * stacks;
    volatile unsigned int * stacksNumDeletedVertices;
    volatile bool* inq;
    volatile bool* inb;
    volatile bool* inp;

    volatile int* node_2_root;
    volatile int* root;

    volatile int* father;
    volatile int* base;
    volatile int* q;
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
    cudaMalloc((void**) &stacks_d, (minimum + 1) * (vertexNum) * sizeof(int) * numBlocks);
    cudaMalloc((void**) &stacksNumDeletedVertices_d, (minimum + 1) * sizeof(unsigned int) * numBlocks);

    stacks.stacks = stacks_d;
    stacks.stacksNumDeletedVertices = stacksNumDeletedVertices_d;
    stacks.minimum = minimum;

    return stacks;
}


Stacks allocateEdmondsStacks(int vertexNum, int numBlocks, unsigned int minimum){
    Stacks stacks;

    volatile int* forest_nodes_stacks_d;
    volatile unsigned int* stacks_depths_d;

    volatile int* father_d;
    volatile int* base_d;
    volatile int* q_d;


    volatile bool* inq_d;
    volatile bool* inb_d;
    volatile bool* inp_d;

    volatile int* node_2_root_d;
    volatile int* root_d;

    cudaMalloc((void**) &forest_nodes_stacks_d, (vertexNum) * sizeof(int) * numBlocks);


    cudaMalloc((void**) &father_d, (vertexNum) * sizeof(int) * numBlocks);
    cudaMalloc((void**) &base_d, (vertexNum) * sizeof(int) * numBlocks);
    cudaMalloc((void**) &q_d, (vertexNum) * sizeof(int) * numBlocks);

    cudaMalloc((void**) &stacks_depths_d, sizeof(unsigned int) * numBlocks);
    cudaMalloc((void**) &inq_d, sizeof(* inq_d) * (vertexNum));
    cudaMalloc((void**) &node_2_root_d, sizeof(*node_2_root_d) * (vertexNum));
    cudaMalloc((void**) &root_d, sizeof(* root_d) * (vertexNum));

    stacks.stacks = forest_nodes_stacks_d;
    stacks.stacksNumDeletedVertices = stacks_depths_d;

    stacks.father = father_d;
    stacks.base = base_d;
    stacks.q = q_d;

    stacks.inq = inq_d;
    stacks.inb = inb_d;
    stacks.inp = inp_d;

    stacks.node_2_root = node_2_root_d;
    stacks.root = root_d;

    stacks.minimum = (vertexNum);

    return stacks;
}


void cudaFreeStacks(Stacks stacks){
    cudaFree((void*)stacks.stacks);
    cudaFree((void*)stacks.stacksNumDeletedVertices);
}

#endif