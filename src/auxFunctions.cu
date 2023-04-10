#include "auxFunctions.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <cstring>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio> 
struct printf_functor { __host__ __device__ void operator()(int x) { printf("%d ", x); } };

// kernel function
template <typename T>
__global__ void setNumInArray(T *arrays, T *index, T *value, int num_index)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid > num_index && value[tid]>0)
        return;
    arrays[index[tid]] = value[tid];
}

// 6V + 4E mem
CSRGraph createCSRGraphFromFile_memopt(const char *filename)
{

	CSRGraph graph;
	int vertexNum;
	int edgeNum;

	FILE *fp;
	fp = fopen(filename, "r");

	int result = fscanf(fp, "%u%u", &vertexNum, &edgeNum);

	//graph.create_memopt(vertexNum, edgeNum);
	thrust::host_vector<unsigned int> keys_h(2*edgeNum);
	thrust::host_vector<unsigned int> values_h(2*edgeNum);
	thrust::host_vector<unsigned int> ones_h(2*edgeNum, 1);

	thrust::host_vector<unsigned int> offsets_h(vertexNum+1);

	thrust::host_vector<unsigned int> keylabel_h(vertexNum);
	thrust::host_vector<unsigned int> nonzerodegrees_h(vertexNum);
	thrust::host_vector<unsigned int> degrees_h(vertexNum);



	int readNum = 0;
	for (int i = 0; i < edgeNum; i++)
	{
		int v0, v1;
		int result = fscanf(fp, "%u%u", &v0, &v1);

		//edgeList[0][i] = v0-1;
		//edgeList[1][i] = v1-1;
		keys_h[readNum] = v0;
		values_h[readNum] = v1;
		++readNum;
		keys_h[readNum] = v1;
		values_h[readNum] = v0;
		++readNum;
	}

	fclose(fp);

	thrust::device_vector<unsigned int> keys_d(2*edgeNum);
	// This will be the dst ptr array.
	thrust::device_vector<unsigned int> values_d(2*edgeNum);
	thrust::device_vector<unsigned int> ones_d(2*edgeNum, 1);

	thrust::device_vector<unsigned int> offsets_d(vertexNum+1);

	thrust::device_vector<unsigned int> keylabel_d(vertexNum);
	thrust::device_vector<unsigned int> nonzerodegrees_d(vertexNum);
	// This will be the degrees array.
	thrust::device_vector<unsigned int> degrees_d(vertexNum);


	keys_d = keys_h;
	values_d = values_h;

	thrust::sort_by_key(thrust::device, keys_d.begin(), keys_d.end(), values_d.begin());

	keys_h = keys_d;
	values_h = values_d;

	thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
	new_end = thrust::reduce_by_key(thrust::device, keys_d.begin(), keys_d.end(), ones_d.begin(), keylabel_d.begin(), nonzerodegrees_d.begin());

	nonzerodegrees_h = nonzerodegrees_d;
	keylabel_h= keylabel_d;

	degrees_h = degrees_d;

    int block_size = 64;
    int num_blocks = (vertexNum + block_size - 1)/block_size;

	unsigned int * degrees_ptr_d = thrust::raw_pointer_cast( degrees_d.data() );
	unsigned int * keylabel_ptr_d = thrust::raw_pointer_cast( keylabel_d.data() );
	unsigned int * nonzerodegrees_ptr_d = thrust::raw_pointer_cast( nonzerodegrees_d.data() );
    setNumInArray<unsigned int> <<<num_blocks, block_size >>> (degrees_ptr_d, keylabel_ptr_d, nonzerodegrees_ptr_d, vertexNum);
	nonzerodegrees_h = nonzerodegrees_d;
	keylabel_h= keylabel_d;

	degrees_h = degrees_d;
	printf("\ni nzd kl d\n");

	for (int i = 0; i < vertexNum; ++i){
		printf(" %d %d %d %d \n", i, nonzerodegrees_h[i], keylabel_h[i],degrees_h[i]);
	}

	keys_h.clear();
	keys_d.clear();
	keylabel_h.clear();
	keylabel_d.clear();
	nonzerodegrees_d.clear();
	nonzerodegrees_h.clear();

	thrust::inclusive_scan(thrust::device, degrees_d.begin(), degrees_d.end(), offsets_d.begin()+1); // in-place scan
	offsets_h = offsets_d;
	printf("offsets\n");

	for (int i = 0; i < vertexNum+1; ++i){
		printf(" %d %d %d \n", i, offsets_h[i], degrees_h[i]);
	}

	printf("DONE\n");
	exit(1);
	return graph;
}


