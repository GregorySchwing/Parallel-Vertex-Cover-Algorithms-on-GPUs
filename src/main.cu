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
#include "ThrustGraph.h"

using namespace std;

#include "matching/match.h"
#include "boostmcm.h"
#include "bfstdcsc.h"
//#include <PCSR.h>


inline void checkLastErrorCUDA(const char *file, int line)
{
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}


int main(int argc, char *argv[]) {

    Config config = parseArgs(argc,argv);
    printf("\nGraph file: %s",config.graphFileName);
    printf("\nUUID: %s\n",config.outputFilePrefix);


    ThrustGraph graph2 = createCSRGraphFromFile_memopt(config.graphFileName);
    CSRGraph graph;
    //graph.create(graph2.vertexNum, graph2.edgeNum);
    // Allocate GPU graph
    CSRGraph graph4m_d;// = allocateGraph2(graph);

    //createfromThrustGraph(graph,graph2);
    //createGPUfromThrustGraph(graph4m_d,graph2);

    //performChecks(graph, config);

    // Allocate GPU graph
    //CSRGraph graph4m_d = allocateGraph(graph);
    struct match mm;
    int exec_policy = 1;
    create_match(graph2.vertexNum,&mm,exec_policy);
    maxmatch_thrust(graph2,&mm,exec_policy);
    exit(1);
    add_edges_to_unmatched_from_last_vertex(graph, graph4m_d, &mm, exec_policy);
    unsigned long ref_size = create_mcm(graph);
    printf("\rgpu mm starting size %lu ref size %lu\n", mm.match_count_h/2, ref_size);


    chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max, elapsed_seconds_edge, elapsed_seconds_mvc;

    //PCSR *pcsr = new PCSR(graph.vertexNum, graph.vertexNum, true, -1);

    //unsigned int minimum = (RemoveMaxMinimum < RemoveEdgeMinimum) ? RemoveMaxMinimum : RemoveEdgeMinimum;
    unsigned int minimum = mm.match_count_h+2;
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

        //printResults(config, RemoveMaxMinimum, RemoveEdgeMinimum, elapsed_seconds_max.count(), elapsed_seconds_edge.count(), minimum, 
        //    elapsed_seconds_mvc.count(), graph.vertexNum, graph.edgeNum, kFound);

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


        setBlockDimAndUseGlobalMemory_DFS(config,graph,maxSharedMemPerMultiProcessor,prop.totalGlobalMem, maxThreadsPerMultiProcessor, maxThreadsPerBlock, 
            maxThreadsPerMultiProcessor, numOfMultiProcessors, minimum);

        //setBlockDimAndUseGlobalMemory(config,graph,maxSharedMemPerMultiProcessor,prop.totalGlobalMem, maxThreadsPerMultiProcessor, maxThreadsPerBlock, 
        //    maxThreadsPerMultiProcessor, numOfMultiProcessors, minimum);
        performChecks(graph, config);

        printf("\nOur Config :\n");
        int numThreadsPerBlock = config.blockDim;
        int numBlocksPerSm; 
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
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkListParameterized_shared_kernel, numThreadsPerBlock, 0);
            } else if(config.version == HYBRID && config.instance==MVC) {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, GlobalWorkList_shared_kernel, numThreadsPerBlock, 0);
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

        // Allocate GPU stack
        Stacks stacks_d;
        stacks_d = allocateStacks(graph.vertexNum,numBlocks,minimum);

        //Global Entries Memory Allocation
        int * global_memory_d;
        if(config.useGlobalMemory){
            cudaMalloc((void**)&global_memory_d, sizeof(int)*graph.vertexNum*numBlocks*2);
        }

        unsigned int * minimum_d;
        unsigned int * solution_mutex_d;
        cudaMalloc((void**) &minimum_d, sizeof(unsigned int));
        cudaMalloc((void**) &solution_mutex_d, sizeof(unsigned int));

        // Allocate counter for each block
        Counters* counters_d;
        cudaMalloc((void**)&counters_d, numBlocks*sizeof(Counters));

        // Copy minimum
        cudaMemcpy(minimum_d, &minimum, sizeof(unsigned int), cudaMemcpyHostToDevice);

        cudaMemset(minimum_d, 0, sizeof(unsigned int));
        cudaMemset(solution_mutex_d, 0, sizeof(unsigned int));

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
            workList_d =  allocateWorkList_DFS(graph, config, numBlocks);    
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
        
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaEvent_t start, stop;
        cudaDeviceSynchronize();
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        Counter counter_h;
        int workListCount_h = 0;
        unsigned int workList_size_h = config.globalListSize;
        HT Pos_h, iter=0;
        FILE *fp;  
        fp = fopen("log.txt", "w");//opening file  
        if (config.useGlobalMemory){
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                //GlobalWorkList_global_kernel <<< numBlocks , numThreadsPerBlock, 0, stream1 >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, global_memory_d, NODES_PER_SM_d);
                GlobalWorkList_global_DFS_kernel <<< numBlocks , numThreadsPerBlock, 0, stream1 >>> (stacks_d, minimum_d, solution_mutex_d, workList_d, graph4m_d, counters_d, first_to_dequeue_global_d, global_memory_d, NODES_PER_SM_d);

               // CSQ returns
                // cudaSuccess if done == 0
                // cudaErrorNotReady if not done == 600
                cudaError_t running = cudaStreamQuery(stream1);
                printf("STATUS %d\n",running);
                //GlobalWorkList_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
                Counter counter_h;
                int workListCount_h = 0;
                cudaError_t error = cudaGetLastError();   // add this line, and check the error code
                while(cudaStreamQuery(stream1)){
                    cudaMemcpyAsync(&counter_h, workList_d.counter, sizeof(Counter), cudaMemcpyDeviceToHost, stream2);
                    cudaStreamSynchronize(stream2);
                    fprintf(fp,"%lu %d %d %d %d\n",iter++,
                                           counter_h.numEnqueued, 
                                           counter_h.numWaiting, 
                                           counter_h.combined);
                }

            } else if(config.version == STACK_ONLY && config.instance==PVC){
                LocalStacksParameterized_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, global_memory_d, k_d, kFound_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                LocalStacks_global_kernel <<< numBlocks , numThreadsPerBlock >>> (stacks_d, graph_d, minimum_d, global_memory_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            }
        } else {
            if (config.version == HYBRID && config.instance==PVC){
                GlobalWorkListParameterized_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, k_d, kFound_d, NODES_PER_SM_d);
            } else if(config.version == HYBRID && config.instance==MVC) {
                GlobalWorkList_shared_DFS_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded, stream1 >>> (stacks_d, minimum_d,  solution_mutex_d,workList_d, graph4m_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
               // CSQ returns
                // cudaSuccess if done == 0
                // cudaErrorNotReady if not done == 600
                cudaError_t running = cudaStreamQuery(stream1);
                printf("STATUS %d\n",running);
                //GlobalWorkList_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, minimum_d, workList_d, graph_d, counters_d, first_to_dequeue_global_d, NODES_PER_SM_d);
                Counter counter_h;
                int workListCount_h = 0;
                cudaError_t error = cudaGetLastError();   // add this line, and check the error code
                while(cudaStreamQuery(stream1)){
                    cudaMemcpyAsync(&counter_h, workList_d.counter, sizeof(Counter), cudaMemcpyDeviceToHost, stream2);
                    cudaMemcpyAsync(&workListCount_h, workList_d.count, sizeof(int), cudaMemcpyDeviceToHost, stream2);
                    cudaMemcpyAsync(&counter_h, workList_d.counter, sizeof(Counter), cudaMemcpyDeviceToHost, stream2);
                    cudaMemcpyAsync(&workListCount_h, workList_d.count, sizeof(int), cudaMemcpyDeviceToHost, stream2);
                    cudaMemcpyAsync(&Pos_h, workList_d.head_tail, sizeof(HT), cudaMemcpyDeviceToHost, stream2);

                    cudaStreamSynchronize(stream2);
                    fprintf(fp,"%lu %d %d %d %d\n",iter++,
                                           Pos_h%workList_size_h, 
                                           counter_h.numEnqueued, 
                                           counter_h.numWaiting, 
                                           counter_h.combined);
                }
            } else if(config.version == STACK_ONLY && config.instance==PVC){
                LocalStacksParameterized_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, graph_d, k_d, kFound_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            } else if(config.version == STACK_ONLY && config.instance==MVC) {
                LocalStacks_shared_kernel <<< numBlocks , numThreadsPerBlock, sharedMemNeeded >>> (stacks_d, graph_d, minimum_d, counters_d, pathCounter_d, NODES_PER_SM_d, config.startingDepth);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaDeviceSynchronize();
        checkLastErrorCUDA(__FILE__, __LINE__);
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

        //printResults(config, RemoveMaxMinimum, RemoveEdgeMinimum, elapsed_seconds_max.count(), elapsed_seconds_edge.count(), minimum, milliseconds, numBlocks, 
        //    numBlocksPerSm, numThreadsPerSM, graph.vertexNum-1, graph.edgeNum, kFound);

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
            cudaFreeWorkList(workList_d);
            cudaFree(first_to_dequeue_global_d);
        }else{
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