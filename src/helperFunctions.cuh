#ifndef HELPFUNC_H
#define HELPFUNC_H

#include "config.h"
#include "bitmask.cuh"

__device__ long long int square(int num){
    return num*num;
}

__device__ bool binarySearch(unsigned int * arr, unsigned int l, unsigned int r, unsigned int x) {
    while (l <= r) {
        unsigned int m = l + (r - l) / 2;
  
        if (arr[m] == x)
            return  true;
  
        if (arr[m] < x)
            l = m + 1;
  
        else
            r = m - 1;
    }
  
    return false;
}

__device__ void deleteNeighborsOfMaxDegreeVertex(CSRGraph graph,int* vertexDegrees_s, unsigned int* numDeletedVertices, int* vertexDegrees_s2, 
    unsigned int* numDeletedVertices2, int maxDegree, unsigned int maxVertex){

    *numDeletedVertices2 = *numDeletedVertices;
    for(unsigned int vertex = threadIdx.x; vertex<graph.vertexNum; vertex+=blockDim.x){
        vertexDegrees_s2[vertex] = vertexDegrees_s[vertex];
    }
    __syncthreads();

    for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1]; edge+=blockDim.x) { // Delete Neighbors of maxVertex
        unsigned int neighbor = graph.dst[edge];
        if (vertexDegrees_s2[neighbor] != -1){
            for(unsigned int neighborEdge = graph.srcPtr[neighbor]; neighborEdge < graph.srcPtr[neighbor + 1]; ++neighborEdge) {
                unsigned int neighborOfNeighbor = graph.dst[neighborEdge];
                if(vertexDegrees_s2[neighborOfNeighbor] != -1) {
                    atomicSub(&vertexDegrees_s2[neighborOfNeighbor], 1);
                }
            }
        }
    }
    
    *numDeletedVertices2 += maxDegree;
    __syncthreads();

    for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1] ; edge += blockDim.x) {
        unsigned int neighbor = graph.dst[edge];
        vertexDegrees_s2[neighbor] = -1;
    }
}

__device__ void deleteMaxDegreeVertex(CSRGraph graph,int* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int maxVertex){

    if(threadIdx.x == 0){
        vertexDegrees_s[maxVertex] = -1;
    }
    ++(*numDeletedVertices);

    __syncthreads(); 

    for(unsigned int edge = graph.srcPtr[maxVertex] + threadIdx.x; edge < graph.srcPtr[maxVertex + 1]; edge += blockDim.x) {
        unsigned int neighbor = graph.dst[edge];
        if(vertexDegrees_s[neighbor] != -1) {
            --vertexDegrees_s[neighbor];
        }
    }
}

__device__ unsigned int leafReductionRule(unsigned int vertexNum, int *vertexDegrees_s, CSRGraph graph, int* shared_mem){

    __shared__ unsigned int numberDeleted;
    __shared__ bool graphHasChanged;

    volatile int * markedForDeletion = shared_mem;
    if (threadIdx.x==0){
        numberDeleted = 0;
    }
    
    do{
        volatile int * vertexDegrees_v = vertexDegrees_s;
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex += blockDim.x){
            markedForDeletion[vertex] = 0;
        }
        if (threadIdx.x==0){
            graphHasChanged = false;
        }
        
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            int degree = vertexDegrees_v[vertex];
            if (degree == 1){
                for(unsigned int edge = graph.srcPtr[vertex] ; edge < graph.srcPtr[vertex + 1]; ++edge) {
                    unsigned int neighbor = graph.dst[edge];
                    int neighborDegree = vertexDegrees_v[neighbor];
                    if(neighborDegree>0){
                        if(neighborDegree > 1 || (neighborDegree == 1 && neighbor<vertex)) {
                            markedForDeletion[neighbor]=1;
                        }
                        break;
                    }
                }
            }
        }
        
        __syncthreads();
        
        for (unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                graphHasChanged = true;
                vertexDegrees_v[vertex] = -1;
                atomicAdd(&numberDeleted,1);
            }
        }
        
        __syncthreads();
        
        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                for(unsigned int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex + 1]; edge++){
                    unsigned int neighbor = graph.dst[edge];
                    if (vertexDegrees_v[neighbor] != -1){
                        atomicSub(&vertexDegrees_s[neighbor],1);
                    }
                }
            }
        }
    
    }while(graphHasChanged);

    __syncthreads();

    return numberDeleted;
}


__device__ unsigned int highDegreeReductionRule(unsigned int vertexNum, int *vertexDegrees_s, CSRGraph graph, int * shared_mem
    , unsigned int numDeletedVertices, unsigned int minimum){

    __shared__ unsigned int numberDeleted_s;
    __shared__ bool graphHasChanged;

    volatile int * markedForDeletion = shared_mem;
    if (threadIdx.x==0){
        numberDeleted_s = 0;
    }

    do{
        volatile int * vertexDegrees_v = vertexDegrees_s;
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex += blockDim.x){
            markedForDeletion[vertex] = 0;
        }
        if (threadIdx.x==0){
            graphHasChanged = false;
        }
        
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            int degree = vertexDegrees_v[vertex];
            if (degree > 0 && degree + numDeletedVertices + numberDeleted_s >= minimum){
                markedForDeletion[vertex]=1;
            }
        }
        
        __syncthreads();
        
        for (unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                graphHasChanged = true;
                vertexDegrees_v[vertex] = -1;
                atomicAdd(&numberDeleted_s,1);
            }
        }
        
        __syncthreads();
                    
        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                for(unsigned int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex + 1]; edge++){
                    unsigned int neighbor = graph.dst[edge];
                    if (vertexDegrees_v[neighbor] != -1){
                        atomicSub(&vertexDegrees_s[neighbor],1);
                    }
                }
            }
        }
    
    }while(graphHasChanged);

    __syncthreads();

    return numberDeleted_s;
}


__device__ unsigned int triangleReductionRule(unsigned int vertexNum, int *vertexDegrees_s, CSRGraph graph, int* shared_mem){

    __shared__ unsigned int numberDeleted;
    __shared__ bool graphHasChanged;

    volatile int * markedForDeletion = shared_mem;
    if (threadIdx.x==0){
        numberDeleted = 0;
    }
    
    do{
        volatile int * vertexDegrees_v = vertexDegrees_s;
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex += blockDim.x){
            markedForDeletion[vertex] = 0;
        }
        if (threadIdx.x==0){
            graphHasChanged = false;
        }
        
        __syncthreads();

        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            int degree = vertexDegrees_v[vertex];
            if (degree == 2){
                unsigned int neighbor1, neighbor2;
                bool foundNeighbor1 = false, keepNeighbors = false;
                for(unsigned int edge = graph.srcPtr[vertex] ; edge < graph.srcPtr[vertex + 1]; ++edge) {
                    unsigned int neighbor = graph.dst[edge];
                    int neighborDegree = vertexDegrees_v[neighbor];
                    if(neighborDegree>0){
                        if(neighborDegree == 1 || neighborDegree == 2 && neighbor < vertex){
                            keepNeighbors = true;
                            break;
                        } else if(!foundNeighbor1){
                            foundNeighbor1 = true;
                            neighbor1 = neighbor;
                        } else {
                            neighbor2 = neighbor;    
                            break;
                        }
                    }
                }

                if(!keepNeighbors){
                    bool found = binarySearch(graph.dst, graph.srcPtr[neighbor2], graph.srcPtr[neighbor2 + 1] - 1, neighbor1);

                    if(found){
                        // Triangle Found
                        markedForDeletion[neighbor1] = true;
                        markedForDeletion[neighbor2] = true;
                        break;
                    }
                }
            }
        }
        
        __syncthreads();
        
        for (unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                graphHasChanged = true;
                vertexDegrees_v[vertex] = -1;
                atomicAdd(&numberDeleted,1);
            }
        }
        
        __syncthreads();
        
        for (unsigned int vertex = threadIdx.x ; vertex < graph.vertexNum; vertex+=blockDim.x){
            if(markedForDeletion[vertex]){ 
                for(unsigned int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex + 1]; edge++){
                    unsigned int neighbor = graph.dst[edge];
                    if (vertexDegrees_v[neighbor] != -1){
                        atomicSub(&vertexDegrees_s[neighbor],1);
                    }
                }
            }
        }
    
    }while(graphHasChanged);

    __syncthreads();

    return numberDeleted;
}


__device__ void findMaxDegree(unsigned int vertexNum, unsigned int *maxVertex, int *maxDegree, int *vertexDegrees_s, int * shared_mem) {
    *maxVertex = 0;
    *maxDegree = 0;
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        int degree = vertexDegrees_s[vertex];
        if(degree > *maxDegree){ 
            *maxVertex = vertex;
            *maxDegree = degree;
        }
    }

    // Reduce max degree
    int * vertex_s = shared_mem;
    int * degree_s = &shared_mem[blockDim.x];
    __syncthreads(); 

    vertex_s[threadIdx.x] = *maxVertex;
    degree_s[threadIdx.x] = *maxDegree;
    __syncthreads();

    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            if(degree_s[threadIdx.x] < degree_s[threadIdx.x + stride]){
                degree_s[threadIdx.x] = degree_s[threadIdx.x + stride];
                vertex_s[threadIdx.x] = vertex_s[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    *maxVertex = vertex_s[0];
    *maxDegree = degree_s[0];
}



__device__ void findUnmatchedNeighbor(CSRGraph graph,unsigned int startingVertex, unsigned int *edgeIndex, unsigned int *neighbor, unsigned int *foundNeighbor,int *visited_s) {
    // All threads perform same work.  It would be possible to use warp shuffles to load
    // the warp with all the neighbors and find the first valid neighbor.
    // This is the key - use the first valid neighbor, that way the linear
    // counter can be used to generate all search trees.
    if (threadIdx.x==0){
        unsigned int start = graph.srcPtr[startingVertex];
        unsigned int end = graph.srcPtr[startingVertex + 1];
        
        //printf("BID %d edgeIndex %d startingVertex %d\n", blockIdx.x,*edgeIndex, startingVertex);
        for(; *edgeIndex < end-start; (*edgeIndex)++) { // Delete Neighbors of startingVertex
            if (graph.matching[startingVertex]>-1 && !visited_s[graph.matching[startingVertex]]){
                    *neighbor = graph.matching[startingVertex];
                    printf("bid %d matched edge %d->%d \n", blockIdx.x,startingVertex, *neighbor);
                    *foundNeighbor=1;
                    break;
            } else {
                if (!visited_s[graph.dst[start + *edgeIndex]]){
                    *neighbor = graph.dst[start + *edgeIndex];
                    *foundNeighbor=2;
                    printf("bid %d unmatched edge %d->%d \n", blockIdx.x,startingVertex, *neighbor);
                    break;
                }
            }
        }
    }
    __syncthreads();
}



__device__ void findUnmatchedNeighbor_bits(CSRGraph graph,unsigned int startingVertex, unsigned int *edgeIndex, unsigned int *neighbor, unsigned int *foundNeighbor,uint32_t *visited_s) {
    // All threads perform same work.  It would be possible to use warp shuffles to load
    // the warp with all the neighbors and find the first valid neighbor.
    // This is the key - use the first valid neighbor, that way the linear
    // counter can be used to generate all search trees.
    if (threadIdx.x==0){
        unsigned int start = graph.srcPtr[startingVertex];
        unsigned int end = graph.srcPtr[startingVertex + 1];
        
        //printf("BID %d edgeIndex %d startingVertex %d\n", blockIdx.x,*edgeIndex, startingVertex);
        for(; *edgeIndex < end-start; (*edgeIndex)++) { // Delete Neighbors of startingVertex
            if (graph.matching[startingVertex]>-1 && !checkVertex(graph.matching[startingVertex],visited_s)){
                    *neighbor = graph.matching[startingVertex];
                    printf("bid %d matched edge %d->%d \n", blockIdx.x,startingVertex, *neighbor);
                    *foundNeighbor=1;
                    break;
            } else {
                if (!checkVertex(graph.dst[start + *edgeIndex],visited_s)){
                    *neighbor = graph.dst[start + *edgeIndex];
                    *foundNeighbor=2;
                    printf("bid %d unmatched edge %d->%d \n", blockIdx.x,startingVertex, *neighbor);
                    break;
                }
            }
        }
    }
    __syncthreads();
}

__device__ void prepareRightChild(CSRGraph graph,int* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int * edgeIndex, int* vertexDegrees_s2, 
    unsigned int* numDeletedVertices2, unsigned int * edgeIndex2){
    if (threadIdx.x==0){
        *numDeletedVertices2 = *numDeletedVertices;
        *edgeIndex2 = *edgeIndex;
    }
    for(unsigned int vertex = threadIdx.x; vertex<graph.vertexNum; vertex+=blockDim.x){
        vertexDegrees_s2[vertex] = vertexDegrees_s[vertex];
    }

    __syncthreads();

    if (threadIdx.x==0){
        *edgeIndex2+=1;
    }
    __syncthreads();

}


__device__ void prepareRightChild_bits(CSRGraph graph,uint32_t* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int * edgeIndex, uint32_t* vertexDegrees_s2, 
    unsigned int* numDeletedVertices2, unsigned int * edgeIndex2){
    if (threadIdx.x==0){
        *numDeletedVertices2 = *numDeletedVertices;
        *edgeIndex2 = *edgeIndex;
    }
    for(unsigned int vertex = threadIdx.x; vertex<(graph.vertexNum+31)/32; vertex+=blockDim.x){
        vertexDegrees_s2[vertex] = vertexDegrees_s[vertex];
    }

    __syncthreads();

    if (threadIdx.x==0){
        *edgeIndex2+=1;
    }
    __syncthreads();

}

__device__ void prepareLeftChild(int* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int* edgeIndex, unsigned int* neighbor){

    if (threadIdx.x==0){
        vertexDegrees_s[*neighbor]=1;
        *numDeletedVertices=*neighbor;
        *edgeIndex=0;
    }

    __syncthreads();

}


__device__ void prepareLeftChild_bits(uint32_t* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int* edgeIndex, unsigned int* neighbor){

    if (threadIdx.x==0){
        setVertex(*neighbor,vertexDegrees_s);
        *numDeletedVertices=*neighbor;
        *edgeIndex=0;
    }

    __syncthreads();

}


__device__ unsigned int findNumOfEdges(unsigned int vertexNum, int *vertexDegrees_s, int * shared_mem){
    int sumDegree = 0;
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        int degree = vertexDegrees_s[vertex];
        if(degree > 0){ 
            sumDegree += degree;
        }
    }
    __syncthreads();
    int * degree_s = shared_mem;
    degree_s[threadIdx.x] = sumDegree;
    
    __syncthreads();
    
    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            degree_s[threadIdx.x] += degree_s[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return degree_s[0]/2;
}

#endif