#ifndef CSRGraph_H
#define CSRGraph_H
#include <stdint.h>

struct DSU{
    int* link;
    int *directParent;
    int *size;
    int *groupRoot;
};

struct CSRGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    int *matching;
    unsigned int *num_unmatched_vertices;
    unsigned int *unmatched_vertices;

    int * oddlvl;
    int * evenlvl;
    bool* pred;
    bool* bridges;
    char* edgeStatus;
    unsigned int *largestVertexChecked;
    uint64_t* bridgeList;
    unsigned int *bridgeList_counter;

    bool* removed;
    int* bud;

    int* visited;

    struct DSU dsu;

    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};


#endif