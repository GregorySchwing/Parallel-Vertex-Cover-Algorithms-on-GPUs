#ifndef CSRGraph_H
#define CSRGraph_H
// For uint64_t
#include <stdint.h>

struct DSU{
    int* link;
    int *directParent;
    int *size;
    int *groupRoot;

    void reset(int n);
    int find(int a);
    int operator[](const int& a);
    void linkTo(int a, int b);

};

struct CSRGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int numBlocks;
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    int *matching;
    unsigned int *num_unmatched_vertices;
    unsigned int *unmatched_vertices;

    int * oddlvl;
    int * evenlvl;
    bool* pred;
    int* bridgeTenacity;
    char* edgeStatus;
    unsigned int *bridgeFront;
    uint64_t* bridgeList;
    unsigned int *bridgeList_counter;

    // DDFS variables
    int * stack1;
    int * stack2;
    int * support;
    int * color;
    int * childsInDDFSTree_keys;
    uint64_t * childsInDDFSTree_values;
    int * ddfsPredecessorsPtr;
    __int128_t * myBridge;
    bool* foundPath;
    int * removedVerticesQueue;
    int * removedPredecessorsSize;

    unsigned int * stack1Top;
    unsigned int * stack2Top; 
    unsigned int * supportTop; 
    unsigned int * globalColorCounter; 
    unsigned int * childsInDDFSTreeTop;
    unsigned int * removedVerticesQueueBack;
    unsigned int * removedVerticesQueueFront;

    bool* removed;

    int* visited;

    struct DSU bud;

    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};


#endif