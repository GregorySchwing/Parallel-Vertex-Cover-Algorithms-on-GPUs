#ifndef HELPFUNC_H
#define HELPFUNC_H

#include "config.h"
__constant__ int MultiplyDeBruijnBitPosition[32] = 
{
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

#define FULL_MASK 0xffffffff

__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }
__device__ inline int lane_id(void) { return threadIdx.x % warpSize; }
__device__ inline int warp_id(void) { return threadIdx.x / warpSize; }

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


__device__ unsigned int binarySearchSrc(unsigned int * arr, unsigned int l, unsigned int r, unsigned int x) {
    // Whether the seach fails or succeeds, depends on whether the edge index of the dst 
    // being searched for is the first edge in the offset of a src
    // In the case of success or failure, the right index will be the index of the
    // value immediately smaller than searched value
    // in the src (offsets) this will be the src containing this edge
    while (l <= r) {
        unsigned int m = l + (r - l) / 2;
  
        if (arr[m] == x)
            break;
  
        if (arr[m] < x)
            l = m + 1;
  
        else
            r = m - 1;
    }
  
    return r;
}

/*
__device__ unsigned int findNextEdge(unsigned int* srcPtrUncompressed, unsigned int *dst, int *vertexDegrees_s, unsigned int currentEdgeIndex, unsigned int numberOfEdges, unsigned int numberOfVertices) {
    while (currentEdgeIndex+threadIdx.x < ((numberOfEdges / warpSize) + 1) * warpSize) {
        int source = srcPtrUncompressed[min(currentEdgeIndex+threadIdx.x, numberOfEdges-1)];
        int destination = dst[min(currentEdgeIndex+threadIdx.x, numberOfEdges-1)];
        //printf("SRC %d DST %d numVertices %d\n",source, destination, numberOfVertices);
        assert(source<numberOfVertices);
        assert(destination<numberOfVertices);

        int sourceDegree = vertexDegrees_s[source];
        int destinationDegree = vertexDegrees_s[destination];

        bool foundEdge = (sourceDegree > 0 && destinationDegree > 0);
        // Use warp-level ballot to check if any thread found an edge
        int warpResult = __ballot_sync(0xffffffff,foundEdge);
        // Use MultiplyDeBruijnBitPosition to get the least significant thread with a true predicate
        if (warpResult)
            return currentEdgeIndex+MultiplyDeBruijnBitPosition[((uint32_t)((warpResult & -warpResult) * 0x077CB531U)) >> 27];
        currentEdgeIndex+=blockDim.x;
    }
    return numberOfEdges;
}
*/


__device__ unsigned int findNextEdge(unsigned int* srcPtrUncompressed, unsigned int *dst, int *vertexDegrees_s, unsigned int currentEdgeIndex, unsigned int numberOfEdges, unsigned int numberOfVertices) {
    for (unsigned int nextEdge = currentEdgeIndex; nextEdge < numberOfEdges; ++nextEdge) {
        int source = srcPtrUncompressed[nextEdge];
        int destination = dst[nextEdge];
        //printf("SRC %d DST %d numVertices %d\n",source, destination, numberOfVertices);
        assert(source<numberOfVertices);
        assert(destination<numberOfVertices);

        int sourceDegree = vertexDegrees_s[source];
        int destinationDegree = vertexDegrees_s[destination];

        bool foundEdge = (sourceDegree > 0 && destinationDegree > 0);
        if (foundEdge)
            return nextEdge;
    }
    return numberOfEdges;
}


__device__ unsigned int findNextEdgeByWarp(unsigned int* srcPtrUncompressed, unsigned int *dst, int *vertexDegrees_s, unsigned int currentEdgeIndex, unsigned int numberOfEdges, unsigned int numberOfVertices) {
    int iterations = divup(numberOfEdges-currentEdgeIndex, warpSize);
    unsigned int nextEdge = currentEdgeIndex + threadIdx.x;
    int source,destination,sourceDegree,destinationDegree;
    bool foundEdge;
    // Block size are limited to 1024; 
    // 32 is the max number of warps per block
    // First 32 entries are whether the predicate is true
    // Second 32 are the value
    __shared__ unsigned int minEdgeIdxInWarp[64];
    __shared__ unsigned int foundEdgeInBlock;
    __shared__ unsigned int lowestEligibleEdge;
    // Min tpb is 64 :)
    if (threadIdx.x < 64)
        minEdgeIdxInWarp[threadIdx.x]=0;
    __syncthreads();

    for (int currentIt = 0; currentIt < iterations; ++currentIt){
        source = srcPtrUncompressed[nextEdge%numberOfEdges];
        destination = dst[nextEdge%numberOfEdges];
        sourceDegree = vertexDegrees_s[source];
        destinationDegree = vertexDegrees_s[destination];
        // Prevent wrap around
        foundEdge = (sourceDegree > 0 && destinationDegree > 0 & nextEdge >= currentEdgeIndex);
        // Use warp-level ballot to check if any thread found an edge
        int warpResult = __ballot_sync(0xffffffff,foundEdge);
        // If result is false, lane 0 writes false and its edge
        // If result is true, winning lane writes true and its edge
        if (lane_id()==MultiplyDeBruijnBitPosition[((uint32_t)((warpResult & -warpResult) * 0x077CB531U)) >> 27]){
            minEdgeIdxInWarp[warp_id()]=warpResult;
            minEdgeIdxInWarp[32+warp_id()]=nextEdge;
        }
        // All warps have written to sm
        __syncthreads();
        if (warp_id()==0){
            foundEdge = minEdgeIdxInWarp[lane_id()];
            foundEdgeInBlock = __ballot_sync(0xffffffff,foundEdge);            
            lowestEligibleEdge = minEdgeIdxInWarp[32+MultiplyDeBruijnBitPosition[((uint32_t)((foundEdgeInBlock & -foundEdgeInBlock) * 0x077CB531U)) >> 27]];
        }
        __syncthreads();
        if(foundEdgeInBlock){
            return lowestEligibleEdge;
        } else {
            nextEdge+=blockDim.x;
        }
    }
    return numberOfEdges;
}


__device__ unsigned int findNextVertexByWarp(unsigned int* srcPtrUncompressed, unsigned int *dst, int *vertexDegrees_s, unsigned int currentVertexIndex, unsigned int numberOfVertices) {
    int iterations = divup(numberOfVertices-currentVertexIndex, warpSize);
    unsigned int nextVertex = currentVertexIndex + threadIdx.x;
    int nextVertexDegree;
    bool foundVertex;
    // Block size are limited to 1024; 
    // 32 is the max number of warps per block
    // First 32 entries are whether the predicate is true
    // Second 32 are the value
    __shared__ unsigned int minEdgeIdxInWarp[64];
    __shared__ unsigned int foundVertexInBlock;
    __shared__ unsigned int lowestEligibleVertex;
    // Min tpb is 64 :)
    if (threadIdx.x < 64)
        minEdgeIdxInWarp[threadIdx.x]=0;
    __syncthreads();

    for (int currentIt = 0; currentIt < iterations; ++currentIt){
        // Prevent oob memory access
        nextVertexDegree = vertexDegrees_s[nextVertex%numberOfVertices];
        // Prevent wrap around, also refind the current vertex on defferred matching branch.
        foundVertex = (nextVertexDegree > -1 & nextVertex >= currentVertexIndex);
        // Use warp-level ballot to check if any thread found an edge
        int warpResult = __ballot_sync(0xffffffff,foundVertex);
        // If result is false, lane 0 writes false and its edge
        // If result is true, winning lane writes true and its edge
        if (lane_id()==MultiplyDeBruijnBitPosition[((uint32_t)((warpResult & -warpResult) * 0x077CB531U)) >> 27]){
            minEdgeIdxInWarp[warp_id()]=warpResult;
            minEdgeIdxInWarp[32+warp_id()]=nextVertex;
        }
        // All warps have written to sm
        __syncthreads();
        if (warp_id()==0){
            foundVertex = minEdgeIdxInWarp[lane_id()];
            foundVertexInBlock = __ballot_sync(0xffffffff,foundVertex);            
            lowestEligibleVertex = minEdgeIdxInWarp[32+MultiplyDeBruijnBitPosition[((uint32_t)((foundVertexInBlock & -foundVertexInBlock) * 0x077CB531U)) >> 27]];
        }
        __syncthreads();
        if(foundVertexInBlock){
            return lowestEligibleVertex;
        } else {
            nextVertex+=blockDim.x;
        }
    }
    return numberOfVertices;
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


__device__ void skipEdge(CSRGraph graph,int* vertexDegrees_s, unsigned int* numDeletedVertices, int* vertexDegrees_s2, 
    unsigned int* numDeletedVertices2, unsigned int nextVertex){
    *numDeletedVertices=nextVertex;
    *numDeletedVertices2 = nextVertex;
    for(unsigned int vertex = threadIdx.x; vertex<graph.vertexNum; vertex+=blockDim.x){
        vertexDegrees_s2[vertex] = vertexDegrees_s[vertex];
    }
    __syncthreads();
    if (threadIdx.x==0){
        if (graph.srcPtr[nextVertex]+vertexDegrees_s2[nextVertex]>=graph.srcPtr[nextVertex+1])
            vertexDegrees_s2[nextVertex]=-1;
        else
            vertexDegrees_s2[nextVertex]++;
    }
    __syncthreads();
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


__device__ void deleteNextEdge(CSRGraph graph,int* vertexDegrees_s, unsigned int nextVertex){
    if(threadIdx.x == 0){
        auto colIndex = graph.srcPtr[nextVertex] + vertexDegrees_s[nextVertex];
        auto dst = graph.dst[colIndex];
        vertexDegrees_s[nextVertex]=-1;
        vertexDegrees_s[dst]=-1;
    }
    __syncthreads();
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

// E <= V/2
// This is the upper bound on edges this branch can add to M
__device__ unsigned int remainingUnmatchedVertices(unsigned int vertexNum, int *vertexDegrees_s, int * shared_mem){
    int sumDegree = 0;
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        int degree = vertexDegrees_s[vertex];
        if(degree > 0){ 
            sumDegree += 1;
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

// Edges added to the matching, have endpoints with degree == -1
__device__ unsigned int getMatchedVertices(unsigned int vertexNum, int *vertexDegrees_s, int * shared_mem){
    int sumDegree = 0;
    for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
        int degree = vertexDegrees_s[vertex];
        if(degree <= 0){ 
            sumDegree += 1;
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
    return degree_s[0];
}


#endif