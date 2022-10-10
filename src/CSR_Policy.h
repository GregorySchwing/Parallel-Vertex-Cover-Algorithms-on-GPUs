#ifndef CSR_GRAPH_POLICY_H
#define CSR_GRAPH_POLICY_H

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

int comparator(const void *elem1, const void *elem2)
{
	int f = *((int *)elem1);
	int s = *((int *)elem2);
	if (f > s)
		return 1;
	if (f < s)
		return -1;
	return 0;
}

template <class T>
struct CSR
{
    CSR(FILE * fp, u_int32_t vertexNum, u_int32_t edgeNum);
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    
    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};

template <class T>
CSR<T>::CSR(FILE * fp, u_int32_t vertexNum, u_int32_t edgeNum)
{
    printf("\nConstructed CSR-based Graph\n");

    this->vertexNum = vertexNum;
    this->edgeNum = edgeNum;

    dst = (unsigned int*)malloc(sizeof(unsigned int)*2*edgeNum);
    srcPtr = (unsigned int*)malloc(sizeof(unsigned int)*(vertexNum+1));
    degree = (int*)malloc(sizeof(int)*vertexNum);
    
    for(unsigned int i=0;i<vertexNum;i++){
        degree[i]=0;
    }

    unsigned int **edgeList = (unsigned int **)malloc(sizeof(unsigned int *) * 2);
	edgeList[0] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);
	edgeList[1] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);

	for (unsigned int i = 0; i < edgeNum; i++)
	{
		unsigned int v0, v1;
		fscanf(fp, "%u%u", &v0, &v1);
		edgeList[0][i] = v0;
		edgeList[1][i] = v1;
	}

	fclose(fp);

	// Gets the degrees of vertices
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		degree[edgeList[0][i]]++;
		if (edgeList[1][i] >= vertexNum)
		{
			printf("\n%d\n", edgeList[1][i]);
		}
		assert(edgeList[1][i] < vertexNum);
		degree[edgeList[1][i]]++;
	}
	// Fill srcPtration array
	unsigned int nextIndex = 0;
	unsigned int *srcPtr2 = (unsigned int *)malloc(sizeof(unsigned int) * vertexNum);
	for (int i = 0; i < vertexNum; i++)
	{
		srcPtr[i] = nextIndex;
		srcPtr2[i] = nextIndex;
		nextIndex += degree[i];
	}
	srcPtr[vertexNum] = edgeNum * 2;
	// fill Graph Array
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		assert(srcPtr2[edgeList[0][i]] < 2 * edgeNum);
		dst[srcPtr2[edgeList[0][i]]] = edgeList[1][i];
		srcPtr2[edgeList[0][i]]++;
		assert(edgeList[1][i] < vertexNum);
		assert(srcPtr2[edgeList[1][i]] < 2 * edgeNum);
		dst[srcPtr2[edgeList[1][i]]] = edgeList[0][i];
		srcPtr2[edgeList[1][i]]++;
	}

	free(srcPtr2);
	free(edgeList[0]);
	edgeList[0] = NULL;
	free(edgeList[1]);
	edgeList[1] = NULL;
	free(edgeList);
	edgeList = NULL;

	for (unsigned int vertex = 0; vertex < vertexNum; ++vertex)
	{
		qsort(&dst[srcPtr[vertex]], degree[vertex], sizeof(int), comparator);
	}
}


template <class T>
void CSR<T>::printGraph(){
        cout<<"\nDegree Array : ";
        for(unsigned int i=0;i<vertexNum;i++){
            cout<<degree[i]<<" ";
        }

        cout<<"\nsrcPtration Array : ";
        for(unsigned int i=0;i<vertexNum;i++){
            cout<<srcPtr[i]<<" ";
        }

        cout<<"\nCGraph Array : ";
        for(unsigned int i=0;i<edgeNum*2;i++){
            cout<<dst[i]<<" ";
        }
        cout<<"\n";
    }

#endif