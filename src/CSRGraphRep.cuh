#include "CSRGraphRep.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
const int INF = 1e9;

void DSU::reset(int n){
    thrust::device_ptr<int> size_thrust_ptr=thrust::device_pointer_cast(size);
    thrust::device_ptr<int> directParent_thrust_ptr=thrust::device_pointer_cast(directParent);
    thrust::device_ptr<int> link_thrust_ptr=thrust::device_pointer_cast(link);
    thrust::device_ptr<int> groupRoot_thrust_ptr=thrust::device_pointer_cast(groupRoot);

    thrust::fill(size_thrust_ptr, size_thrust_ptr+n, 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_ptr, directParent_thrust_ptr+n, -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_ptr,link_thrust_ptr+n, 0, 1);
    thrust::sequence(groupRoot_thrust_ptr,groupRoot_thrust_ptr+n, 0, 1);
}
__device__ int DSU::find(int a){
    return link[a] = (a == link[a] ? a : find(link[a]));
}
__device__ int DSU::operator[](const int& a){
    return groupRoot[find(a)];
}
__device__ void DSU::linkTo(int a, int b){
    assert(directParent[a] == -1);
    assert(directParent[b] == -1);
    directParent[a] = b;
    a = find(a);
    b = find(b);
    int gr = groupRoot[b];
    assert(a != b);
    
    if(size[a] > size[b])
        std::swap(a,b);
    link[b] = a;
    size[a] += size[b];
    groupRoot[a] = gr;
}

DSU allocate_DSU(struct CSRGraph graph){

    DSU dsu;
    cudaMalloc((void**) &dsu.directParent,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &dsu.groupRoot,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &dsu.link,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &dsu.size,sizeof(int)*graph.vertexNum);

    thrust::device_ptr<int> size_thrust_ptr=thrust::device_pointer_cast(dsu.size);
    thrust::device_ptr<int> directParent_thrust_ptr=thrust::device_pointer_cast(dsu.directParent);
    thrust::device_ptr<int> link_thrust_ptr=thrust::device_pointer_cast(dsu.link);
    thrust::device_ptr<int> groupRoot_thrust_ptr=thrust::device_pointer_cast(dsu.groupRoot);

    thrust::fill(size_thrust_ptr, size_thrust_ptr+graph.vertexNum, 1); // or 999999.f if you prefer
    thrust::fill(directParent_thrust_ptr, directParent_thrust_ptr+graph.vertexNum, -1); // or 999999.f if you prefer

    thrust::sequence(link_thrust_ptr,link_thrust_ptr+graph.vertexNum, 0, 1);
    thrust::sequence(groupRoot_thrust_ptr,groupRoot_thrust_ptr+graph.vertexNum, 0, 1);

    return dsu;
}


CSRGraph allocateGraph(CSRGraph graph){
    CSRGraph Graph;

    unsigned int* dst_d;
    unsigned int* srcPtr_d;
    int* degree_d;
    int *matching_d;
    unsigned int *num_unmatched_vertices_d;
    unsigned int *unmatched_vertices_d;

    int * oddlvl_d;
    int * evenlvl_d;
    bool* pred_d;
    bool* bridges_d;
    char* edgeStatus_d;
    uint64_t* bridgeList_d;
    unsigned int *bridgeList_counter_d;

    bool * removed_d;
    unsigned int *bridgeFront;
    cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum);
    cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1));
    cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum);
    cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*(graph.vertexNum));
    cudaMalloc((void**) &num_unmatched_vertices_d,sizeof(unsigned int));

    cudaMalloc((void**) &pred_d, 2*graph.edgeNum*sizeof(bool));
    cudaMalloc((void**) &bridges_d, 2*graph.edgeNum*sizeof(bool));
    cudaMalloc((void**) &edgeStatus_d, 2*graph.edgeNum*sizeof(char));

    cudaMalloc((void**) &oddlvl_d, graph.vertexNum*sizeof(int));
    cudaMalloc((void**) &evenlvl_d, graph.vertexNum*sizeof(int));

    cudaMalloc((void**) &bridgeFront,sizeof(unsigned int));
    cudaMalloc((void**) &removed_d, graph.vertexNum*sizeof(bool));

    // Likely too much work.
    cudaMalloc((void**) &bridgeList_counter_d,sizeof(unsigned int));
    cudaMalloc((void**) &bridgeList_d,2*graph.edgeNum*sizeof(uint64_t));
    cudaMemset(bridgeList_counter_d, 0, sizeof(unsigned int));
    cudaMemset(bridgeFront, 0, sizeof(unsigned int));
    Graph.bridgeList=bridgeList_d;
    Graph.bridgeList_counter=bridgeList_counter_d;
    
    /*
    cudaMemset(oddlvl_d, INF, graph.vertexNum*sizeof(int));
    cudaMemset(evenlvl_d, INF, graph.vertexNum*sizeof(int));

    */

    thrust::device_ptr<int> oddlvl_thrust_ptr=thrust::device_pointer_cast(oddlvl_d);
    thrust::device_ptr<int> evenlvl_thrust_ptr=thrust::device_pointer_cast(evenlvl_d);

    thrust::fill(oddlvl_thrust_ptr, oddlvl_thrust_ptr+graph.vertexNum, INF); // or 999999.f if you prefer
    thrust::fill(evenlvl_thrust_ptr, evenlvl_thrust_ptr+graph.vertexNum, INF); // or 999999.f if you prefer
    cudaMemset(bridgeFront, 0, sizeof(unsigned int));
    cudaMemset(matching_d, -1, graph.vertexNum*sizeof(int));

    Graph.vertexNum = graph.vertexNum;
    Graph.edgeNum = graph.edgeNum;
    //Graph.num_unmatched_vertices = graph.num_unmatched_vertices;
    Graph.num_unmatched_vertices = num_unmatched_vertices_d;
    Graph.dst = dst_d;
    Graph.srcPtr = srcPtr_d;
    Graph.degree = degree_d;
    Graph.matching = matching_d;
    Graph.unmatched_vertices = unmatched_vertices_d;

    
    Graph.oddlvl = oddlvl_d;
    Graph.evenlvl = evenlvl_d;
    Graph.pred = pred_d;
    Graph.bridges = bridges_d;
    Graph.edgeStatus = edgeStatus_d;
    Graph.bridgeFront = bridgeFront;
    Graph.removed=removed_d;
    Graph.bud = allocate_DSU(graph);

    cudaMemcpy(dst_d,graph.dst,sizeof(unsigned int)*2*graph.edgeNum,cudaMemcpyHostToDevice);
    cudaMemcpy(srcPtr_d,graph.srcPtr,sizeof(unsigned int)*(graph.vertexNum+1),cudaMemcpyHostToDevice);
    cudaMemcpy(degree_d,graph.degree,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);
    //cudaMemcpy(unmatched_vertices_d,graph.unmatched_vertices,sizeof(unsigned int)*graph.num_unmatched_vertices,cudaMemcpyHostToDevice);
    //cudaMemcpy(matching_d,graph.matching,sizeof(int)*graph.vertexNum,cudaMemcpyHostToDevice);

    return Graph;
}

void cudaFreeGraph(CSRGraph graph){
    cudaFree(graph.dst);
    cudaFree(graph.srcPtr);
    cudaFree(graph.degree);
    if (graph.matching != NULL)
    cudaFree(graph.matching);
    if (graph.unmatched_vertices != NULL)
    cudaFree(graph.unmatched_vertices);
}