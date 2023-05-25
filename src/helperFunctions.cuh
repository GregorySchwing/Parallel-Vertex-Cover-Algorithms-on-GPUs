#ifndef HELPFUNC_H
#define HELPFUNC_H

#include "config.h"
enum EdgeType {NotScanned, Prop, Bridge};

template <typename T> __device__ void inline swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}


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

__device__ void setLvl(CSRGraph & graph, int u, int lev){
    if(lev&1) graph.oddlvl[u] = lev; else graph.evenlvl[u] = lev;
}

__device__ int minlvl(CSRGraph & graph, int u){
    return min(graph.oddlvl[u], graph.evenlvl[u]);
}

__device__ int tenacity(CSRGraph & graph, int u, int v){
    if(graph.matching[u] == v)
        return graph.oddlvl[u] + graph.oddlvl[v] + 1;
    return graph.evenlvl[u] + graph.evenlvl[v] + 1;
}

__device__ void removeAndPushToQueue(CSRGraph & graph, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u) {graph.removed[u] = true; removedVerticesQueue[removedVerticesQueueBack[0]++]=u;}

__device__ void flip(CSRGraph & graph, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u, int v) {
    if(graph.removed[u] || graph.removed[v] || graph.matching[u] == v) return;//flipping only unmatched edges
    removeAndPushToQueue(graph,removedVerticesQueue,removedVerticesQueueBack,u);
    removeAndPushToQueue(graph,removedVerticesQueue,removedVerticesQueueBack,v);
    graph.matching[u] = v;
    graph.matching[v] = u;
}
__device__ bool openingDfs(CSRGraph & graph, int * color, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int cur, int bcur, int b);
__device__ void augumentPath(CSRGraph & graph, int * color, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u, int v, bool initial=false);
__device__ void augumentPath(CSRGraph & graph, int * color, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int u, int v, bool initial){
    if(u == v) return;
    if(!initial && minlvl(graph,u) == graph.evenlvl[u]) { //simply follow predecessors
        // TMP
        int x = 0;
        /*
        assert(predecessors[u].size() == 1); //u should be evenlevel (last minlevel edge is matched, so there is only one predecessor)
        int x = predecessors[u][0]; //no need to flip edge since we know it's matched

        int idx = 0;
        while(graph.bud[predecessors[x][idx]] != graph.bud[x]) {
            idx++; 
            assert(idx < (int)predecessors[x].size());
        }
        u = predecessors[x][idx];
        */
        assert(!graph.removed[u]);
        flip(graph,removedVerticesQueue,removedVerticesQueueBack,x,u);
        augumentPath(graph,color,removedVerticesQueue,removedVerticesQueueBack,u,v);
    }
    else { //through bridge
        int u3 = graph.myBridge[u] >> 96; 
        int v3 = graph.myBridge[u] >> 64; 
        int u2 = graph.myBridge[u] >> 32; 
        int v2 = (int32_t)graph.myBridge[u];
        if((color[u2]^1) == color[u] || color[v2] == color[u]) {
            swap(u2, v2);
            swap(u3,v3);
        }

        flip(graph,removedVerticesQueue,removedVerticesQueueBack,u3,v3);
        bool openingDfsSucceed1 = openingDfs(graph,color,removedVerticesQueue,removedVerticesQueueBack,u3,u2,u);
        assert(openingDfsSucceed1);

        int v4 = graph.bud.directParent[u];
        bool openingDfsSucceed2 = openingDfs(graph,color,removedVerticesQueue,removedVerticesQueueBack,v3,v2,v4);
        assert(openingDfsSucceed2);
        augumentPath(graph,color,removedVerticesQueue,removedVerticesQueueBack,v4,v);
    }
}

__device__ bool openingDfs(CSRGraph & graph, int * color, int * removedVerticesQueue, unsigned int * removedVerticesQueueBack, int cur, int bcur, int b){
    if(bcur == b) {
        augumentPath(graph,color,removedVerticesQueue,removedVerticesQueueBack,cur,bcur);
        return true;
    }
    /*
    for(auto a: childsInDDFSTree[bcur]) { 
        if((a.nd == b || color[a.nd] == color[bcur]) && openingDfs(a.st,a.nd,b)) {
            augumentPath(graph,color,removedVerticesQueue,removedVerticesQueueBack,cur,bcur);
            flip(graph,removedVerticesQueue,removedVerticesQueueBack,bcur,a >> 32);
            return true;
        }
    }
    */
    return false;
}


/*
tries to move color1 down, updating colors, stacks and childs in ddfs tree
also adds each visited vertex to support of this bridge
*/

__device__ int ddfsMove(CSRGraph & graph, int * stack1, int * stack2, unsigned int * stack1Top, unsigned int * stack2Top, int * support, unsigned int * supportTop, int * color, unsigned int *globalColorCounter, int * ddfsPredecessorsPtr, int*budAtDDFSEncounter, const int color1, const int color2) {
//int ddfsMove(vector<int>& stack1, const int color1, vector<int>& stack2, const int color2, vector<int>& support) {
    
    //int u = stack1.back();
    int u = stack1[stack1Top[0]];
    unsigned int start = graph.srcPtr[u];
    unsigned int end = graph.srcPtr[u + 1];
    unsigned int edgeIndex;
    for(edgeIndex=start+ddfsPredecessorsPtr[u]; edgeIndex < end-start; edgeIndex++) { // Delete Neighbors of startingVertex
        if (graph.pred[edgeIndex]) {
            ddfsPredecessorsPtr[u]=edgeIndex+1;
            int a = graph.dst[edgeIndex];
            int v = graph.bud[a];
            assert(graph.removed[a] == graph.removed[v]);
            if(graph.removed[a])
                continue;
            if(color[v] == 0) {
                //stack1.push_back(v);
                stack1[stack1Top[0]++]=v;
                //support.push_back(v);
                support[supportTop[0]++]=v;

                //childsInDDFSTree[u].push_back({a,v});
                budAtDDFSEncounter[u]=v;
                //childsInDDFSTree_values[childsInDDFSTreeTop[0]]=(uint64_t) a << 32 | v;
                color[v] = color1;
                return -1;
            }
            else if(v == stack2[stack2Top[0]]){
                budAtDDFSEncounter[u]=v;
                //childsInDDFSTree_values[childsInDDFSTreeTop[0]]=(uint64_t) a << 32 | v;
            }
        }
        --stack1Top[0];

        if(stack1Top[0] == 0) {
            if(stack2Top[0] == 1) { //found bottleneck
                color[stack2[stack2Top[0]]] = 0;
                return stack2[stack2Top[0]];
            }
            //change colors
            assert(color[stack2[stack2Top[0]]] == color2);
            stack1[stack1Top[0]++]=stack2[stack2Top[0]];
            color[stack1[stack1Top[0]]] = color1;
            --stack2Top[0];
        }
    }

    return -1;
}




//returns {r0, g0} or {bottleneck, bottleneck} packed into uint64_t
__device__ uint64_t ddfs(CSRGraph & graph, int src, int dst, int * stack1, int * stack2, unsigned int * stack1Top, unsigned int * stack2Top, int * support, unsigned int * supportTop, int * color, unsigned int *globalColorCounter, int * ddfsPredecessorsPtr, int*budAtDDFSEncounter) {
    stack1[stack1Top[0]++]=graph.bud[src];
    stack2[stack2Top[0]++]=graph.bud[dst];
    //vector<int> Sr = {graph.bud[src]}, Sg = {graph.bud[dst]};
    //if(Sr[0] == Sg[0])
    //    return {Sr[0],Sg[0]};
    if (stack1[0]==stack2[0]){
        return (uint64_t) stack1[0] << 32 | stack2[0];
    }
    //out_support = {Sr[0], Sg[0]};
    support[supportTop[0]++]=stack1[0];
    support[supportTop[0]++]=stack2[0];

    //int newRed = color[Sr[0]] = ++globalColorCounter, newGreen = color[Sg[0]] = ++globalColorCounter;
    //assert(newRed == (newGreen^1));
    int newRed = color[stack1[0]] = ++globalColorCounter[0], newGreen = color[stack2[0]] = ++globalColorCounter[0];
    assert(newRed == (newGreen^1));
    

    for(;;) {
        //if found two disjoint paths
        //if(minlvl(Sr.back()) == 0 && minlvl(Sg.back()) == 0)
        if(minlvl(graph,stack1[stack1Top[0]]) == 0 && minlvl(graph,stack2[stack2Top[0]]) == 0)
            //return {Sr.back(),Sg.back()};
            return (uint64_t) stack1[stack1Top[0]] << 32 | stack2[stack2Top[0]];

        int b;
        //if(minlvl(Sr.back()) >= minlvl(Sg.back()))
        if(minlvl(graph,stack1[stack1Top[0]]) >= minlvl(graph,stack2[stack2Top[0]]))
            //b = ddfsMove(Sr,newRed,Sg, newGreen, out_support);
            b = ddfsMove(graph,stack1,stack2,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,ddfsPredecessorsPtr,budAtDDFSEncounter, newGreen, newRed);
        else
            //b = ddfsMove(Sg,newGreen,Sr, newRed, out_support);
            b = ddfsMove(graph,stack2,stack1,stack1Top,stack2Top,support,supportTop,color,globalColorCounter,ddfsPredecessorsPtr,budAtDDFSEncounter, newRed, newGreen);
        if(b != -1)
            //return {b,b};
            return (uint64_t) b << 32 | b;

    }
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

__device__ void prepareLeftChild(int* vertexDegrees_s, unsigned int* numDeletedVertices, unsigned int* edgeIndex, unsigned int* neighbor){

    if (threadIdx.x==0){
        vertexDegrees_s[*neighbor]=1;
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