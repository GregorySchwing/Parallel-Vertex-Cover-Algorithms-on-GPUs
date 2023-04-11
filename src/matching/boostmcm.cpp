#include "boostmcm.h"
#include "bfstdcsc.h"

unsigned long create_mcm(CSRGraph & graph){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost(graph.dst,graph.srcPtr,graph.vertexNum);
}
unsigned long create_mcm(ThrustGraph & graph){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    unsigned int *CP_h= thrust::raw_pointer_cast(graph.offsets_h.data() );
    unsigned int *IC_h  = thrust::raw_pointer_cast( graph.values_h.data() );
    return mcm_boost(IC_h,CP_h,graph.vertexNum);
}
unsigned long create_mcm_headstart(ThrustGraph & graph,struct match * mm){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost_headstart(graph.dst,graph.srcPtr,mm->m_h,graph.vertexNum);
}
unsigned long create_mcm_headstart(CSRGraph & graph,CSRGraph & graph_d,struct match * mm){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost_headstart(graph.dst,graph.srcPtr,mm->m_h,graph.vertexNum);
}
