#include <chrono> 
#include <time.h>
#include <math.h>
#include <stdexcept>
#include "config.h"
#include "stack.cuh"
#include "Sequential.h"
#include "auxFunctions.h"
#include "CSRGraphRep.cuh"
#define USE_GLOBAL_MEMORY 0
#include "LocalStacks.cuh"
#include "GlobalWorkList.cuh"
#include "LocalStacksParameterized.cuh"
#include "GlobalWorkListParameterized.cuh"
#undef USE_GLOBAL_MEMORY
#define USE_GLOBAL_MEMORY 1
#include "LocalStacks.cuh"
#include "GlobalWorkList.cuh"
#include "LocalStacksParameterized.cuh"
#include "GlobalWorkListParameterized.cuh"
#undef USE_GLOBAL_MEMORY
#include "SequentialParameterized.h"
#include "KernelArgs.cuh"

#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>
using namespace std;



int main(int argc, char *argv[]) {
    KernelArgs fbArgs;

    Config config = parseArgs(argc,argv);
    printf("\nGraph file: %s",config.graphFileName);
    printf("\nUUID: %s\n",config.outputFilePrefix);

    CSRGraph graph = createCSRGraphFromFile(config.graphFileName);
    performChecks(graph, config);

    chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max, elapsed_seconds_edge, elapsed_seconds_mvc;

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

    unsigned int k = config.k; 
    unsigned int kFound = 0;

    if(config.version == SEQUENTIAL){
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

        setBlockDimAndUseGlobalMemory(config,graph,maxSharedMemPerMultiProcessor,prop.totalGlobalMem, maxThreadsPerMultiProcessor, maxThreadsPerBlock, 
            maxThreadsPerMultiProcessor, numOfMultiProcessors, minimum);
        performChecks(graph, config);

        printf("\nOur Config :\n");
        int numThreadsPerBlock = config.blockDim;
        int numBlocksPerSm; 

        int sharedMemNeeded = graph.vertexNum;
        if(graph.vertexNum > numThreadsPerBlock*2){
            sharedMemNeeded+=graph.vertexNum;
        } else {
            sharedMemNeeded+=numThreadsPerBlock*2;
        }
        sharedMemNeeded *= sizeof(int);

        if (config.useGlobalMemory){
            if (config.version == HYBRID && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkListParameterized_global_kernel, numThreadsPerBlock, 0);
            } else if(config.version == HYBRID && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkList_global_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacksParameterized_global_kernel, numThreadsPerBlock, 0);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacks_global_kernel, numThreadsPerBlock, 0);
            }
        } else {
            if (config.version == HYBRID && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkListParameterized_shared_kernel, numThreadsPerBlock, sharedMemNeeded);
            } else if(config.version == HYBRID && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkList_shared_kernel, numThreadsPerBlock, sharedMemNeeded);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacksParameterized_shared_kernel, numThreadsPerBlock, sharedMemNeeded);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, LocalStacks_shared_kernel, numThreadsPerBlock, sharedMemNeeded);
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
        CSRGraph graph_d;
        fbArgs.graph = allocateGraph(graph);

        // Allocate GPU stack
        Stacks stacks_d;
        fbArgs.stacks = allocateStacks(graph.vertexNum,numBlocks,minimum);

        //Global Entries Memory Allocation
        int * global_memory_d;
        if(config.useGlobalMemory){
            cudaMalloc((void**)&global_memory_d, sizeof(int)*graph.vertexNum*numBlocks*2);
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
            fbArgs.workList =  allocateWorkList(graph, config, numBlocks);    
        } else {
            cudaMalloc((void**)&pathCounter_d, sizeof(unsigned int));
            cudaMemcpy(pathCounter_d, &pathCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
        
        int supportsCoopLaunch = 0;
        int dev = 0;
        cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev); 
        if( supportsCoopLaunch != 1)
            throw std::runtime_error("Cooperative Launch is not supported on this machine configuration.");

        cudaEvent_t start, stop;
        cudaDeviceSynchronize();
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        fbArgs.minimum=minimum_d;
        fbArgs.counters=counters_d;
        fbArgs.first_to_dequeue_global=first_to_dequeue_global_d;
        fbArgs.NODES_PER_SM=NODES_PER_SM_d;
        if(config.useGlobalMemory){
            fbArgs.global_memory=global_memory_d;
            sharedMemNeeded=0;
        }
        void *kernel_args[] = {&fbArgs};
        if (config.useGlobalMemory){
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                //GlobalWorkList_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, NODES_PER_SM_d);
                cudaLaunchCooperativeKernel((void *)GlobalWorkList_global_kernel, numBlocks, numThreadsPerBlock, (void **) (&kernel_args), sharedMemNeeded);
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                LocalStacksParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, global_memory_d, k_d, kFound_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                LocalStacks_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, minimum_d, global_memory_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            }
        } else {
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                //GlobalWorkList_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
                cudaLaunchCooperativeKernel((void *)GlobalWorkList_shared_kernel, numBlocks, numThreadsPerBlock, (void **) (&kernel_args), sharedMemNeeded);
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
        cudaFreeGraph(fbArgs.graph);

        cudaFreeStacks(fbArgs.stacks);
        
        #if USE_GLOBAL_MEMORY
        cudaFree(global_memory_d);
    
        #endif

        if(config.version == HYBRID){
            cudaFreeWorkList(fbArgs.workList);
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
        printf("\nSize of minimum vertex cover: %u\n\n", minimum);
    }

    return 0;
}