#ifndef CSRGraph_H
#define CSRGraph_H


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

struct CSRGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    int *matching;
    unsigned int *num_unmatched_vertices;
    unsigned int *unmatched_vertices;

    int* visited;

	thrust::host_vector<unsigned int> keys_h;
	thrust::host_vector<unsigned int> values_h;
	thrust::host_vector<unsigned int> ones_h;

	thrust::host_vector<unsigned int> offsets_h;

	thrust::host_vector<unsigned int> keylabel_h;
	thrust::host_vector<unsigned int> nonzerodegrees_h;
	thrust::host_vector<int> degrees_h;


	thrust::device_vector<unsigned int> keys_d;
	// This will be the dst ptr array.
	thrust::device_vector<unsigned int> values_d;
	thrust::device_vector<unsigned int> ones_d;

	thrust::device_vector<unsigned int> offsets_d;

	thrust::device_vector<unsigned int> keylabel_d;
	thrust::device_vector<unsigned int> nonzerodegrees_d;
	// This will be the degrees array.
	thrust::device_vector<int> degrees_d;
    void create_memopt(unsigned int xn,unsigned int xm); // Initializes the graph rep

    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};


#endif