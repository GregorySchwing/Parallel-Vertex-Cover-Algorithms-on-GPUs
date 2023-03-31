#include "boostmcm.h"

unsigned long create_mcm(struct Graph * graph){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost(graph->CscA.IC,graph->CscA.CP,graph->N);
}
unsigned long create_mcm_headstart(struct Graph * graph,struct match * mm){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_boost_headstart(graph->CscA.IC,graph->CscA.CP,mm->m_h,graph->N);
}
