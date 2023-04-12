#include "edmonds.h"

unsigned long create_mcm_edmonds(struct Graph * graph,struct match * mm){       
    // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...
    return mcm_edmonds(graph->CscA.IC,graph->CscA.CP,mm->m_h,graph->N);
    //return 0;
}