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
Graph<GraphPolicy>::Graph(FILE * fp, uint32_t vertexNum, uint32_t edgeNum):GraphPolicy(vertexNum, edgeNum){	

    // add some edges
    //for (int i = 0; i < edgeNum; i++) {
    //GraphPolicy::add_edge(i, i, 1);
    //}

	for (unsigned int i = 0; i < edgeNum; i++)
	{
		unsigned int v0, v1;
		fscanf(fp, "%u%u", &v0, &v1);
        //printf("adding edge %u %u", v0, v1);
        GraphPolicy::add_edge(v0, v1, 1);
	}

	fclose(fp);

    // update the values of some edges

    //for (int i = 0; i < 5; i++) {
    //GraphPolicy::add_edge_update(i, i, 2);
    //}

    // print out the graph
    //GraphPolicy::print_graph();

}

#endif