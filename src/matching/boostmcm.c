#include "boostmcm.h"

unsigned long create_mcm(CSRGraph & graph){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost(graph.dst,graph.srcPtr,graph.vertexNum);
}
unsigned long create_mcm_headstart(CSRGraph & graph,CSRGraph & graph_d,struct match * mm){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost_headstart(graph.dst,graph.srcPtr,mm->m_h,graph.vertexNum);
}
