#ifndef ThrustGraph_H
#define ThrustGraph_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>


struct ThrustGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges

    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    int *matching;
    unsigned int *num_unmatched_vertices;
    unsigned int *unmatched_vertices;


	thrust::host_vector<unsigned int> values_h;
	thrust::host_vector<unsigned int> offsets_h;
	thrust::host_vector<int> degrees_h;
	thrust::host_vector<int> matching_h;
	thrust::host_vector<int> unmatched_vertices_h;
	thrust::host_vector<int> num_unmatched_vertices_h;


	thrust::device_vector<unsigned int> offsets_d;
	// This will be the dst ptr array.
	thrust::device_vector<unsigned int> values_d;
	// This will be the degrees array.
	thrust::device_vector<int> degrees_d;
	thrust::device_vector<int> matching_d;
	thrust::device_vector<int> unmatched_vertices_d;
	thrust::device_vector<int> num_unmatched_vertices_d;


	thrust::host_vector<unsigned int> keys_h;
	thrust::host_vector<unsigned int> ones_h;
	thrust::host_vector<unsigned int> keylabel_h;
	thrust::host_vector<unsigned int> nonzerodegrees_h;

	thrust::device_vector<unsigned int> keys_d;
	thrust::device_vector<unsigned int> ones_d;
	thrust::device_vector<unsigned int> keylabel_d;
	thrust::device_vector<unsigned int> nonzerodegrees_d;


};

ThrustGraph createCSRGraphFromFile_memopt(const char *filename);

#endif