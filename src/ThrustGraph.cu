#include "ThrustGraph.h"


// kernel function
template <typename T>
__global__ void setNumInArray(int *arrays, T *index, T *value, int num_index)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_index)
        return;
    arrays[index[tid]] = value[tid];
}

// 6V + 4E mem
ThrustGraph createCSRGraphFromFile_memopt(const char *filename)
{

	ThrustGraph graph;
	int vertexNum;
	int edgeNum;

	FILE *fp;
	fp = fopen(filename, "r");

	int result = fscanf(fp, "%u%u", &vertexNum, &edgeNum);
	graph.vertexNum = vertexNum;
	graph.edgeNum = edgeNum;
	printf("vertexNum %d, edgeNum %d\n",vertexNum,edgeNum);
	//graph.create_memopt(vertexNum, edgeNum);
	graph.keys_h.resize(2*edgeNum);
	graph.values_h.resize(2*edgeNum);
	graph.ones_h.resize(2*edgeNum, 1);

	graph.offsets_h.resize(vertexNum+1);

	graph.keylabel_h.resize(vertexNum);
	graph.nonzerodegrees_h.resize(vertexNum);
	graph.degrees_h.resize(vertexNum);



	int readNum = 0;
	for (int i = 0; i < edgeNum; i++)
	{
		int v0, v1;
		int result = fscanf(fp, "%u%u", &v0, &v1);

		//edgeList[0][i] = v0-1;
		//edgeList[1][i] = v1-1;
		graph.keys_h[readNum] = v0;
		graph.values_h[readNum] = v1;
		++readNum;
		graph.keys_h[readNum] = v1;
		graph.values_h[readNum] = v0;
		++readNum;
	}

	fclose(fp);

	graph.keys_d.resize(2*edgeNum);
	// This will be the dst ptr array.
	graph.values_d.resize(2*edgeNum);
	graph.ones_d.resize(2*edgeNum, 1);

	graph.offsets_d.resize(vertexNum+1);

	graph.keylabel_d.resize(vertexNum);
	graph.nonzerodegrees_d.resize(vertexNum);
	// This will be the degrees array.
	graph.degrees_d.resize(vertexNum);


	graph.keys_d = graph.keys_h;
	graph.values_d = graph.values_h;

	thrust::sort_by_key(thrust::device, graph.keys_d.begin(), graph.keys_d.end(), graph.values_d.begin());

	graph.keys_h = graph.keys_d;
	graph.values_h = graph.values_d;

	thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
	new_end = thrust::reduce_by_key(thrust::device, graph.keys_d.begin(), graph.keys_d.end(), graph.ones_d.begin(), graph.keylabel_d.begin(), graph.nonzerodegrees_d.begin());

	graph.nonzerodegrees_h = graph.nonzerodegrees_d;
	graph.keylabel_h= graph.keylabel_d;

	graph.degrees_h = graph.degrees_d;

    int block_size = 64;
    int num_blocks = (vertexNum + block_size - 1)/block_size;

	int * degrees_ptr_d = thrust::raw_pointer_cast( graph.degrees_d.data() );
	unsigned int * keylabel_ptr_d = thrust::raw_pointer_cast( graph.keylabel_d.data() );
	unsigned int * nonzerodegrees_ptr_d = thrust::raw_pointer_cast( graph.nonzerodegrees_d.data() );
    setNumInArray<unsigned int> <<<num_blocks, block_size >>> (degrees_ptr_d, keylabel_ptr_d, nonzerodegrees_ptr_d, vertexNum);
	graph.nonzerodegrees_h = graph.nonzerodegrees_d;
	graph.keylabel_h= graph.keylabel_d;

	graph.degrees_h = graph.degrees_d;

	graph.keys_h.clear();
	graph.keys_d.clear();
	graph.keylabel_h.clear();
	graph.keylabel_d.clear();
	graph.nonzerodegrees_d.clear();
	graph.nonzerodegrees_h.clear();

	thrust::inclusive_scan(thrust::device, graph.degrees_d.begin(), graph.degrees_d.end(), graph.offsets_d.begin()+1); // in-place scan
	graph.offsets_h = graph.offsets_d;

	graph.matching_h.resize(vertexNum);
	graph.matching_d.resize(vertexNum);
	graph.unmatched_vertices_h.resize(vertexNum);
	graph.unmatched_vertices_d.resize(vertexNum);
	graph.num_unmatched_vertices_h.resize(1);
	graph.num_unmatched_vertices_d.resize(1);
	printf("DONE\n");
	//graph.create_memopt(vertexNum, edgeNum);
	return graph;
}


