#ifndef SEQ_H
#define SEQ_H

#include "CSRGraphRep.h"
#include "helperFunctions.h"
#include "stack.h"
#include <stdlib.h>
#define NUM_LEVELS 1
unsigned int Sequential(CSRGraph graph, unsigned int minimum);
unsigned int SequentialBFSDFS(CSRGraph graph, unsigned int minimum);
unsigned int SequentialDFSDFS(CSRGraph graph, unsigned int minimum);
unsigned int SequentialDFSDFS_M_ALT(CSRGraph graph, unsigned int minimum);
#endif