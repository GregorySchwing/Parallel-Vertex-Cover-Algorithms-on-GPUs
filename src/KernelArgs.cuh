#ifndef KERNELARGS_H
#define KERNELARGS_H
#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"
#include "CSRGraphRep.cuh"

struct KernelArgs
{
    Stacks stacks;
    unsigned int * minimum;
    WorkList workList; 
    CSRGraph graph;
    Counters* counters;
    int* first_to_dequeue_global; 
    int* global_memory; 
    int* NODES_PER_SM;
};
#endif