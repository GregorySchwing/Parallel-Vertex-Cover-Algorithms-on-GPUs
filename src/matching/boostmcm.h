/*
 *  * Functions to:
 *   1) compute serial mcm using boost
 *
 */

#ifndef BOOSTMCM_H
#define BOOSTMCM_H

//#include "graph.h"
//#include "bfs.h"
#include "CSRGraphRep.h"
struct mcm {
    int *m_h, *m_d;
    int *req_h, *req_d;
};

unsigned long create_mcm(CSRGraph & graph);
unsigned long create_mcm_headstart(CSRGraph & graph,CSRGraph & graph_d,struct match * mm);
#endif