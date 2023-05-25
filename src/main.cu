#include <chrono> 
#include <time.h>
#include <math.h>
#include "config.h"
#include "stack.cuh"
#include "Sequential.h"
#include "auxFunctions.h"
#include "CSRGraphRep.cuh"
#define USE_GLOBAL_MEMORY 0
#include "LocalStacks.cuh"
#include "GlobalWorkList.cuh"
#include "GlobalWorkListDFS.cuh"

#include "LocalStacksParameterized.cuh"
#include "GlobalWorkListParameterized.cuh"
#undef USE_GLOBAL_MEMORY
#define USE_GLOBAL_MEMORY 1
#include "LocalStacks.cuh"
#include "GlobalWorkList.cuh"
#include "GlobalWorkListDFS.cuh"
#include "LocalStacksParameterized.cuh"
#include "GlobalWorkListParameterized.cuh"
#undef USE_GLOBAL_MEMORY
#include "SequentialParameterized.h"
#include "GlobalWorkListBFS.cuh"

using namespace std;
// BFS
#define MAX_THREADS_PER_GRID (2**31)
#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK/THREADS_PER_WARP)
#define I_SIZE ((3/2)*THREADS_PER_BLOCK)

int main(int argc, char *argv[]) {

    Config config = parseArgs(argc,argv);
    printf("\nGraph file: %s",config.graphFileName);
    printf("\nUUID: %s\n",config.outputFilePrefix);

    CSRGraph graph = createCSRGraphFromFile(config.graphFileName);
    for (int i = 0; i < graph.vertexNum; ++i)
        printf ("%d %d %d\n", i, graph.degree[i], graph.srcPtr[i]);
    //performChecks(graph, config);
    chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max, elapsed_seconds_edge, elapsed_seconds_mvc;
    unsigned int RemoveMaxMinimum = 0;
    unsigned int RemoveEdgeMinimum = 0;

    /*
    begin = std::chrono::system_clock::now(); 
    unsigned int RemoveMaxMinimum = RemoveMaxApproximateMVC(graph);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 

    printf("\nElapsed Time for Approximate Remove Max: %f\n",elapsed_seconds_max.count());
    printf("Approximate Remove Max Minimum is: %u\n", RemoveMaxMinimum);
    fflush(stdout);

    begin = std::chrono::system_clock::now();
    unsigned int RemoveEdgeMinimum = RemoveEdgeApproximateMVC(graph);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_edge = end - begin; 

    printf("Elapsed Time for Approximate Remove Edge: %f\n",elapsed_seconds_edge.count());
    printf("Approximate Remove Edge Minimum is: %u\n", RemoveEdgeMinimum);
    fflush(stdout);

    unsigned int minimum = (RemoveMaxMinimum < RemoveEdgeMinimum) ? RemoveMaxMinimum : RemoveEdgeMinimum;
    */
    unsigned int minimum = graph.vertexNum;
    unsigned int k = config.k; 
    unsigned int kFound = 0;
    int result = 0;

    if(config.version == SEQUENTIAL){
        /*
        if(config.instance == PVC){
            begin = std::chrono::system_clock::now();
            minimum = SequentialParameterized(graph, minimum, k, &kFound);
            end = std::chrono::system_clock::now(); 
            elapsed_seconds_mvc = end - begin; 
        } else {
            begin = std::chrono::system_clock::now();
            minimum = Sequential(graph, minimum);
            end = std::chrono::system_clock::now(); 
            elapsed_seconds_mvc = end - begin; 
        } 

        printResults(config, RemoveMaxMinimum, RemoveEdgeMinimum, elapsed_seconds_max.count(), elapsed_seconds_edge.count(), minimum, 
            elapsed_seconds_mvc.count(), graph.vertexNum, graph.edgeNum, kFound);

        printf("\nElapsed time: %fs",elapsed_seconds_mvc.count());
        */
    } else {
        cudaDeviceSynchronize();

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\nDevice name: %s\n\n", prop.name);

        int numOfMultiProcessors;
        cudaDeviceGetAttribute(&numOfMultiProcessors,cudaDevAttrMultiProcessorCount,0);
        printf("NumOfMultiProcessors : %d\n",numOfMultiProcessors);

        int maxThreadsPerMultiProcessor;
        cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor,cudaDevAttrMaxThreadsPerMultiProcessor,0);
        printf("MaxThreadsPerMultiProcessor : %d\n",maxThreadsPerMultiProcessor);

        int maxThreadsPerBlock;
        cudaDeviceGetAttribute(&maxThreadsPerBlock,cudaDevAttrMaxThreadsPerBlock,0);
        printf("MaxThreadsPerBlock : %d\n",maxThreadsPerBlock);

        int maxSharedMemPerMultiProcessor;
        cudaDeviceGetAttribute(&maxSharedMemPerMultiProcessor,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);
        printf("MaxSharedMemPerMultiProcessor : %d\n",maxSharedMemPerMultiProcessor);

        //setBlockDimAndUseGlobalMemory(config,graph,maxSharedMemPerMultiProcessor,prop.totalGlobalMem, maxThreadsPerMultiProcessor, maxThreadsPerBlock, 
        //    maxThreadsPerMultiProcessor, numOfMultiProcessors, minimum);

        setBlockDimAndUseGlobalMemoryDFS_NoStack(config,graph,maxSharedMemPerMultiProcessor,prop.totalGlobalMem, maxThreadsPerMultiProcessor, maxThreadsPerBlock, 
            maxThreadsPerMultiProcessor, numOfMultiProcessors, graph.vertexNum);
        //performChecks(graph, config);

        printf("\nOur Config :\n");
        int numThreadsPerBlock = config.blockDim;
        int numBlocksPerSm; 
        if (config.useGlobalMemory){
            if (config.version == HYBRID && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkListParameterized_global_kernel, numThreadsPerBlock, 0);
            } else if(config.version == HYBRID && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkList_global_DFS_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacksParameterized_global_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacks_global_kernel, numThreadsPerBlock, 0);
            }
        } else {
            if (config.version == HYBRID && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkListParameterized_shared_kernel, numThreadsPerBlock, 0);
            } else if(config.version == HYBRID && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkList_shared_DFS_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacksParameterized_shared_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacks_shared_kernel, numThreadsPerBlock, 0);
            }
        }

        unsigned int tempNumBlocks;
        if(config.numBlocks){
            tempNumBlocks = config.numBlocks;
        } else {
            tempNumBlocks = numBlocksPerSm*numOfMultiProcessors;
        }

        const unsigned int numBlocks = tempNumBlocks;
        int numThreadsPerSM = numBlocksPerSm * numThreadsPerBlock;
        printf("NumOfThreadPerBlocks : %d\n",numThreadsPerBlock);
        printf("NumOfBlocks : %u\n",numBlocks);
        printf("NumOfBlockPerSM : %d\n",numBlocksPerSm);
        printf("NumOfThreadsPerSM : %d\n\n",numThreadsPerSM);
        fflush(stdout);
        graph.numBlocks = numBlocks;
        //Allocate NODES_PER_SM
        int * NODES_PER_SM_d;
        #if USE_COUNTERS
            int * NODES_PER_SM;
            NODES_PER_SM = (int *)malloc(sizeof(int)*numOfMultiProcessors);
            for (unsigned int i = 0;i<numOfMultiProcessors;++i){
                NODES_PER_SM[i]=0;
            }
            cudaMalloc((void**)&NODES_PER_SM_d, numOfMultiProcessors*sizeof(int));
            cudaMemcpy(NODES_PER_SM_d, NODES_PER_SM, numOfMultiProcessors*sizeof(int), cudaMemcpyHostToDevice);
        #endif

        // Allocate GPU graph
        CSRGraph graph_d = allocateGraph(graph);
        DFSWorkList dfsWL_d =  allocateDFSWorkList(graph);
        printf("Allocated graph\n");
        cudaError_t erra = cudaDeviceSynchronize();
        if(erra != cudaSuccess) {
            printf("GPU Error: %s\n", cudaGetErrorString(erra));
            exit(1);
        }
        // Allocate GPU stack
        Stacks stacks_d;
        //stacks_d = allocateStacks(graph.vertexNum,numBlocks,minimum);

        //Global Entries Memory Allocation
        int * global_memory_d;
        if(config.useGlobalMemory){
            //cudaMalloc((void**)&global_memory_d, sizeof(int)*graph.vertexNum*numBlocks*2);
        }

        unsigned int * minimum_d;
        cudaMalloc((void**) &minimum_d, sizeof(unsigned int));

        // Allocate counter for each block
        Counters* counters_d;
        cudaMalloc((void**)&counters_d, numBlocks*sizeof(Counters));

        // Copy minimum
        cudaMemcpy(minimum_d, &minimum, sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int *k_d = NULL;
        unsigned int *kFound_d = NULL;
        if(config.instance == PVC){
            cudaMalloc((void**)&k_d, sizeof(unsigned int));
            cudaMemcpy(k_d, &k, sizeof(unsigned int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&kFound_d, sizeof(unsigned int));
            cudaMemcpy(kFound_d, &kFound, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        // HYBRID
        // Allocate GPU queue
        WorkList workList_d;
        //First to dequeue flag
        int *first_to_dequeue_global_d;
        int first_to_dequeue_global=0;
        // STACKONLY
        unsigned int * pathCounter_d;
        unsigned int pathCounter = 0;
        if(config.version == HYBRID){
            cudaMalloc((void**)&first_to_dequeue_global_d, sizeof(int));
            cudaMemcpy(first_to_dequeue_global_d, &first_to_dequeue_global, sizeof(int), cudaMemcpyHostToDevice);
            //workList_d =  allocateWorkList(graph, config, numBlocks);    
        } else {
            cudaMalloc((void**)&pathCounter_d, sizeof(unsigned int));
            cudaMemcpy(pathCounter_d, &pathCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        int sharedMemNeeded = graph.vertexNum;
        if(graph.vertexNum > numThreadsPerBlock*2){
            sharedMemNeeded+=graph.vertexNum;
        } else {
            sharedMemNeeded+=numThreadsPerBlock*2;
        }
        sharedMemNeeded *= sizeof(int);
        
        cudaEvent_t start, stop;
        cudaDeviceSynchronize();
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        #if USE_GLOBAL_MEMORY
        GlobalDFSKernelArgs args;
        args.global_memory = global_memory_d; 
        #else
        SharedDFSKernelArgs args;
        #endif
        args.stacks = stacks_d; 
        args.minimum = minimum_d; 
        args.workList = workList_d; 
        args.dfsWL = dfsWL_d;
        args.graph = graph_d; 
        args.counters = counters_d; 
        args.first_to_dequeue_global = first_to_dequeue_global_d; 
        args.NODES_PER_SM = NODES_PER_SM_d;
        void *kernel_args[] = {&args};
        printf("Launching kernel\n");
        int dimGridBFS = (graph.vertexNum + THREADS_PER_BLOCK)/THREADS_PER_BLOCK;
        if (config.useGlobalMemory){
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                bool pathFound = false;
                bool pathFoundOnAnyIteration = false;
                int iter = 0;
                do {
                    printf("Iter %d\n",iter++);
                    cudaMemset(&graph_d.foundPath[0], false, sizeof(bool));
                    pathFound = false;
                    pathFoundOnAnyIteration = false;
                    unsigned int depth = 0;
                    GlobalWorkList_Set_Sources_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
                    for (; depth < graph.vertexNum && !pathFound; ++depth){ 
                        GlobalWorkList_BFS_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d, depth);
                        GlobalWorkList_Extract_Bridges_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, dfsWL_d,  graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d, depth);
                        cudaLaunchCooperativeKernel((void*)(GlobalWorkList_global_DFS_kernel), numBlocks, numThreadsPerBlock, kernel_args) ;
                        cudaDeviceSynchronize();
                        cudaMemcpy(&pathFound, &graph_d.foundPath[0], sizeof(bool), cudaMemcpyDeviceToHost);
                        //GlobalWorkList_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, NODES_PER_SM_d);
                        printf("Found path %s\n",pathFound?"True":"False");
                        pathFoundOnAnyIteration |= pathFound;
                    }
                    graph_d.reset();
                    dfsWL_d.reset();
                    //exit(1);
                } while(pathFoundOnAnyIteration);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                LocalStacksParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, global_memory_d, k_d, kFound_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                LocalStacks_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, minimum_d, global_memory_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            }
        } else {
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                bool pathFound = false;
                bool pathFoundOnAnyIteration = false;
                int iter = 0;
                do {
                    printf("Iter %d\n",iter++);
                    cudaMemset(&graph_d.foundPath[0], false, sizeof(bool));
                    pathFound = false;
                    pathFoundOnAnyIteration = false;
                    unsigned int depth = 0;
                    GlobalWorkList_Set_Sources_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
                    for (; depth < graph.vertexNum && !pathFound; ++depth){ 
                        GlobalWorkList_BFS_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d, depth);
                        GlobalWorkList_Extract_Bridges_kernel <<< dimGridBFS, THREADS_PER_BLOCK >>> (stacks_d, minimum_d, workList_d, dfsWL_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d, depth);
                        cudaLaunchCooperativeKernel((void*)(GlobalWorkList_shared_DFS_kernel), numBlocks, numThreadsPerBlock, kernel_args) ;
                        cudaDeviceSynchronize();
                        cudaMemcpy(&pathFound, &graph_d.foundPath[0], sizeof(bool), cudaMemcpyDeviceToHost);
                        //GlobalWorkList_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, NODES_PER_SM_d);
                        printf("Found path %s\n",pathFound?"True":"False");
                        pathFoundOnAnyIteration |= pathFound;
                    }
                    graph_d.reset();
                    dfsWL_d.reset();
                    //exit(1);

                } while(pathFoundOnAnyIteration);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                LocalStacksParameterized_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, graph_d, k_d, kFound_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                LocalStacks_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, graph_d, minimum_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess) {
            printf("GPU Error: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        // Copy back result
        if(config.instance == PVC){
            cudaMemcpy(&kFound, kFound_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(&minimum, minimum_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time: %fms \n", milliseconds);

        printResults(config, RemoveMaxMinimum, RemoveEdgeMinimum, elapsed_seconds_max.count(), elapsed_seconds_edge.count(), minimum, milliseconds, numBlocks, 
            numBlocksPerSm, numThreadsPerSM, graph.vertexNum-1, graph.edgeNum, kFound);
        thrust::device_ptr<int> m_vec=thrust::device_pointer_cast(graph_d.matching);
        using namespace thrust::placeholders;
        result = thrust::count_if(m_vec, m_vec+graph.vertexNum, _1 > -1);
        #if USE_COUNTERS
        printCountersInFile(config,counters_d,numBlocks);
        printNodesPerSM(config,NODES_PER_SM_d,numOfMultiProcessors);
        cudaFree(NODES_PER_SM);
        #endif

        if(config.instance == PVC){
            cudaFree(k_d);
        }
        graph.del();
        cudaFree(minimum_d);
        cudaFree(counters_d);
        cudaFreeGraph(graph_d);

        cudaFreeStacks(stacks_d);
        
        #if USE_GLOBAL_MEMORY
        cudaFree(global_memory_d);
        #endif

        if(config.version == HYBRID){
            //cudaFreeWorkList(workList_d);
            cudaFree(first_to_dequeue_global_d);
        } else {
            cudaFree(pathCounter_d);
        }

    }

    if(config.instance == PVC){
        if(kFound){
            printf("\nMinimum is less than or equal to K: %u\n\n",k);
        } else {
            printf("\nMinimum is greater than K: %u\n\n",k);
        }
    } else {
        printf("\nSize of matching: %u\n\n", result/2);
    }

    return 0;
}