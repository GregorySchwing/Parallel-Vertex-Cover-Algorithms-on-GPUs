#ifndef GRAPH_H
#define GRAPH_H

#include "GraphPolicy.h"

// Library code
template <class GraphPolicy>
class Graph : public GraphPolicy
{
    public:
        Graph(FILE * fp, uint32_t vertexNum, uint32_t edgeNum);
};

template <class GraphPolicy>
Graph<GraphPolicy>::Graph(FILE * fp, uint32_t vertexNum, uint32_t edgeNum):GraphPolicy(fp, vertexNum, edgeNum){	

    GraphPolicy::printGraph();

}

#endif