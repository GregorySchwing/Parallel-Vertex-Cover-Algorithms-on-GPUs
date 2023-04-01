#ifndef SEQUENTIALSTACK_H
#define SEQUENTIALSTACK_H

#include <stdint.h>

struct Stack
{
    bool foundSolution;
    unsigned int size;
    int top;
    int *stack;
    unsigned int *startVertex;
    unsigned int *stackNumDeletedVertices;
    unsigned int *backtrackingIndices;
    void print(int numVertices);
};

#endif