#include "CSRGraphRep.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

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
    checkCudaErrors(cudaMalloc((void**) &dsu.directParent,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &dsu.groupRoot,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &dsu.link,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &dsu.size,sizeof(int)*graph.vertexNum));

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
    int* bridgeTenacity_d;
    char* edgeStatus_d;
    uint64_t* bridgeList_d;
    unsigned int *bridgeList_counter_d;
    unsigned int *bridgeFront_d;


    // DDFS variables
    int * stack1_d;
    int * stack2_d;
    int * support_d;
    int * color_d;
    int * budAtDDFSEncounter_d;
    int * ddfsPredecessorsPtr_d;
    __int128_t * myBridge_d;
    bool* foundPath_d;
    int * removedVerticesQueue_d;
    int * removedPredecessorsSize_d;

    unsigned int * stack1Top_d;
    unsigned int * stack2Top_d; 
    unsigned int * supportTop_d; 
    unsigned int * globalColorCounter_d; 
    unsigned int * removedVerticesQueueBack_d;
    unsigned int * removedVerticesQueueFront_d;

    bool * removed_d;
    checkCudaErrors(cudaMalloc((void**) &dst_d,sizeof(unsigned int)*2*graph.edgeNum));
    checkCudaErrors(cudaMalloc((void**) &srcPtr_d,sizeof(unsigned int)*(graph.vertexNum+1)));
    checkCudaErrors(cudaMalloc((void**) &degree_d,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &matching_d,sizeof(int)*graph.vertexNum));
    checkCudaErrors(cudaMalloc((void**) &unmatched_vertices_d,sizeof(unsigned int)*(graph.vertexNum)));
    checkCudaErrors(cudaMalloc((void**) &num_unmatched_vertices_d,sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void**) &pred_d, 2*graph.edgeNum*sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**) &bridgeTenacity_d, 2*graph.edgeNum*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &edgeStatus_d, 2*graph.edgeNum*sizeof(char)));

    checkCudaErrors(cudaMalloc((void**) &oddlvl_d, graph.vertexNum*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &evenlvl_d, graph.vertexNum*sizeof(int)));

    checkCudaErrors(cudaMalloc((void**) &bridgeFront_d,sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &removed_d, graph.vertexNum*sizeof(bool)));

    // Likely too much work.
    checkCudaErrors(cudaMalloc((void**) &bridgeList_counter_d,sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &bridgeList_d,2*graph.edgeNum*sizeof(uint64_t)));

    cudaMemset(bridgeList_counter_d, 0, sizeof(unsigned int));
    cudaMemset(bridgeFront_d, 0, sizeof(unsigned int));
    Graph.bridgeList=bridgeList_d;
    Graph.bridgeList_counter=bridgeList_counter_d;
    
	// budAtDDFSEncounter<int>[N],
	// stack1<int>[N], 
    // stack2<int>[N], 
	// support<int>[N], 
    // ddfsPredecessorsPtr<int>[N], 
    // color[N]
	// Scalars: stack1Top, stack2Top, globalColorCounter, supportTop
    checkCudaErrors(cudaMalloc((void**) &stack1_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &stack2_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &support_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &color_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &budAtDDFSEncounter_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &ddfsPredecessorsPtr_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &myBridge_d,graph.vertexNum*sizeof(__int128_t)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &foundPath_d,sizeof(bool)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &removedVerticesQueue_d,sizeof(int)*graph.vertexNum*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &removedPredecessorsSize_d,sizeof(int)*graph.vertexNum*graph.numBlocks));

    Graph.stack1=stack1_d;
    Graph.stack2=stack2_d;
    Graph.support=support_d;
    Graph.color=color_d;
    Graph.budAtDDFSEncounter=budAtDDFSEncounter_d;
    Graph.ddfsPredecessorsPtr=ddfsPredecessorsPtr_d;
    Graph.myBridge=myBridge_d;
    Graph.foundPath=foundPath_d;
    Graph.removedVerticesQueue = removedVerticesQueue_d;
    Graph.removedPredecessorsSize=removedPredecessorsSize_d;
    printf("NUM BLOCKS %d\n",graph.numBlocks);
    checkCudaErrors(cudaMalloc((void**) &stack1Top_d,sizeof(unsigned int)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &stack2Top_d,sizeof(unsigned int)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &supportTop_d,sizeof(unsigned int)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &globalColorCounter_d,sizeof(unsigned int)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &removedVerticesQueueBack_d,sizeof(unsigned int)*graph.numBlocks));
    checkCudaErrors(cudaMalloc((void**) &removedVerticesQueueFront_d,sizeof(unsigned int)*graph.numBlocks));

    Graph.stack1Top=stack1Top_d;
    Graph.stack2Top=stack2Top_d;
    Graph.supportTop=supportTop_d;
    Graph.globalColorCounter=globalColorCounter_d;
    Graph.removedVerticesQueueBack=removedVerticesQueueBack_d;
    Graph.removedVerticesQueueFront=removedVerticesQueueFront_d;

    cudaMemset(Graph.stack1Top, 0, sizeof(unsigned int)*graph.numBlocks);
    cudaMemset(Graph.stack2Top, 0, sizeof(unsigned int)*graph.numBlocks);
    cudaMemset(Graph.supportTop, 0, sizeof(unsigned int)*graph.numBlocks);
    cudaMemset(Graph.removedVerticesQueueBack, 0, sizeof(unsigned int)*graph.numBlocks);
    cudaMemset(Graph.removedVerticesQueueFront, 0, sizeof(unsigned int)*graph.numBlocks);

    /*
    cudaMemset(oddlvl_d, INF, graph.vertexNum*sizeof(int));
    cudaMemset(evenlvl_d, INF, graph.vertexNum*sizeof(int));

    */
    thrust::device_ptr<unsigned int> globalColorCounter_thrust_ptr=thrust::device_pointer_cast(Graph.globalColorCounter);
    thrust::device_ptr<int> oddlvl_thrust_ptr=thrust::device_pointer_cast(oddlvl_d);
    thrust::device_ptr<int> evenlvl_thrust_ptr=thrust::device_pointer_cast(evenlvl_d);

    thrust::fill(oddlvl_thrust_ptr, oddlvl_thrust_ptr+graph.vertexNum, INF); // or 999999.f if you prefer
    thrust::fill(evenlvl_thrust_ptr, evenlvl_thrust_ptr+graph.vertexNum, INF); // or 999999.f if you prefer
    thrust::fill(globalColorCounter_thrust_ptr, globalColorCounter_thrust_ptr+graph.numBlocks, 1); // or 999999.f if you prefer

    cudaMemset(matching_d, -1, graph.vertexNum*sizeof(int));
    cudaMemset(edgeStatus_d, 0, 2*graph.edgeNum*sizeof(char));
    cudaMemset(removed_d, 0, graph.vertexNum*sizeof(bool));
    cudaMemset(removedPredecessorsSize_d, 0, graph.numBlocks*graph.vertexNum*sizeof(int));


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
    Graph.bridgeTenacity = bridgeTenacity_d;
    Graph.edgeStatus = edgeStatus_d;
    Graph.bridgeFront = bridgeFront_d;
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