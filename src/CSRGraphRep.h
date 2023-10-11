#ifndef CSRGraph_H
#define CSRGraph_H

struct CSRGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int degreeZeroVertices; // Number of degree zero vertices.
    unsigned int* dst;
    unsigned int* srcPtr;
    unsigned int* srcPtrUncompressed;
    int* degree;
    
    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};


#endif