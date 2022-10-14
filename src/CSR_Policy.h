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
	CSR(const CSR<T> & otherCSR);
    ~CSR();
    unsigned int RemoveMaxApproximateMVC();
    bool leafReductionRule(unsigned int &minimum);
    bool triangleReductionRule(unsigned int &minimum);
    bool auxBinarySearch(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x);
    unsigned int RemoveEdgeApproximateMVC();
    unsigned int getRandom(int lower, int upper);

    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    
    // Moved initialization into construction
	//void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    // Moved this into a copy constructor.
	//void copy(CSR<T> graph);

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
CSR<T>::~CSR()
{
    free(dst);
    dst = NULL;
    free(srcPtr);
    srcPtr = NULL;
    free(degree);
    degree = NULL;
}

template <class T>
unsigned int CSR<T>::RemoveMaxApproximateMVC(){

	CSR<T> approxGraph = (*this);

	unsigned int minimum = 0;
	bool hasEdges = true;
	while (hasEdges)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = approxGraph.leafReductionRule(minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = approxGraph.triangleReductionRule(minimum);
			}
			++iterationCounter;
		} while (triangleHasChanged);

		unsigned int maxV;
		int maxD = 0;
		for (unsigned int i = 0; i < approxGraph.vertexNum; i++)
		{
			if (approxGraph.degree[i] > maxD)
			{
				maxV = i;
				maxD = approxGraph.degree[i];
			}
		}
		if (maxD == 0)
			hasEdges = false;
		else
		{
			approxGraph.deleteVertex(maxV);
			++minimum;
		}
	}

	approxGraph.del();

	return minimum;
}

template <class T>
unsigned int CSR<T>::RemoveEdgeApproximateMVC(){

	CSR<T> approxGraph = (*this);

	unsigned int minimum = 0;
	unsigned int numRemainingEdges = approxGraph.edgeNum;

	for (unsigned int vertex = 0; vertex < approxGraph.vertexNum && numRemainingEdges > 0; vertex++)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = approxGraph.leafReductionRule(minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = approxGraph.triangleReductionRule(minimum);
			}
			++iterationCounter;
		} while (triangleHasChanged);

		if (approxGraph.degree[vertex] > 0)
		{

			unsigned int randomEdge = getRandom(approxGraph.srcPtr[vertex], approxGraph.srcPtr[vertex] + approxGraph.degree[vertex] - 1);

			numRemainingEdges -= approxGraph.degree[vertex];
			numRemainingEdges -= approxGraph.degree[approxGraph.dst[randomEdge]];
			++numRemainingEdges;
			approxGraph.deleteVertex(vertex);
			approxGraph.deleteVertex(approxGraph.dst[randomEdge]);
			minimum += 2;
		}
	}

	approxGraph.del();

	return minimum;
}


template <class T>
unsigned int CSR<T>::getRandom(int lower, int upper)
{
	srand(time(0));
	unsigned int num = (rand() % (upper - lower + 1)) + lower;
	return num;
}

template <class T>
bool CSR<T>::leafReductionRule(unsigned int &minimum){
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < vertexNum; ++i)
		{
			if (degree[i] == 1)
			{
				hasChanged = true;
				for (unsigned int j = srcPtr[i]; j < srcPtr[i + 1]; ++j)
				{
					if (degree[dst[j]] != -1)
					{
						unsigned int neighbor = dst[j];
						deleteVertex(neighbor);
						++minimum;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

template <class T>
bool CSR<T>::triangleReductionRule(unsigned int &minimum)
{
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < vertexNum; ++i)
		{
			if (degree[i] == 2)
			{

				unsigned int neighbor1, neighbor2;
				bool foundNeighbor1 = false, keepNeighbors = false;
				for (unsigned int edge = srcPtr[i]; edge < srcPtr[i + 1]; ++edge)
				{
					unsigned int neighbor = dst[edge];
					int neighborDegree = degree[neighbor];
					if (neighborDegree > 0)
					{
						if (neighborDegree == 1 || neighborDegree == 2 && neighbor < i)
						{
							keepNeighbors = true;
							break;
						}
						else if (!foundNeighbor1)
						{
							foundNeighbor1 = true;
							neighbor1 = neighbor;
						}
						else
						{
							neighbor2 = neighbor;
							break;
						}
					}
				}

				if (!keepNeighbors)
				{
					bool found = auxBinarySearch(dst, srcPtr[neighbor2], srcPtr[neighbor2 + 1] - 1, neighbor1);

					if (found)
					{
						hasChanged = true;
						// Triangle Found
						deleteVertex(neighbor1);
						deleteVertex(neighbor2);
						minimum += 2;
						break;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

template <class T>
bool CSR<T>::auxBinarySearch(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x)
{
	while (l <= r)
	{
		unsigned int m = l + (r - l) / 2;

		if (arr[m] == x)
			return true;

		if (arr[m] < x)
			l = m + 1;

		else
			r = m - 1;
	}

	return false;
}


template <class T>
CSR<T>::CSR(const CSR<T> & otherCSR)
{
    vertexNum =otherCSR.vertexNum;
    edgeNum =otherCSR.edgeNum;

    dst = (unsigned int*)malloc(sizeof(unsigned int)*2*edgeNum);
    srcPtr = (unsigned int*)malloc(sizeof(unsigned int)*(vertexNum+1));
    degree = (int*)malloc(sizeof(int)*vertexNum);

    for(unsigned int i = 0;i<vertexNum;i++){
        srcPtr[i] = otherCSR.srcPtr[i];
        degree[i] = otherCSR.degree[i];
    }
    srcPtr[vertexNum] = otherCSR.srcPtr[vertexNum];

    for(unsigned int i=0;i<2*edgeNum;i++){
        dst[i] = otherCSR.dst[i];
    }
}

template <class T>
void CSR<T>::deleteVertex(unsigned int v){
    assert(v < vertexNum);
    assert(srcPtr[v]+degree[v] <= 2*edgeNum);
    for(int i = srcPtr[v];i<srcPtr[v]+degree[v];i++){
        int neighbor = dst[i];
        assert(neighbor < vertexNum);
        int last = srcPtr[neighbor]+degree[neighbor];
        assert(last <= 2*edgeNum);
        for(int j = srcPtr[neighbor];j<last;j++){
            if(dst[j]==v){
                dst[j] = dst[last-1];
                if(degree[neighbor]!=-1){
                    degree[neighbor]--;
                }
            }
        } 
    }
    degree[v]=-1;
}

template <class T>
unsigned int  CSR<T>::findMaxDegree(){
    unsigned int max = 0;
    unsigned int maxd = 0;
    for(unsigned int i=0;i<vertexNum;i++){
        if(degree[i]>maxd){
            maxd = degree[i];
            max = i;
        }
    }
    return max;
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


template <class T>
void CSR<T>::del(){

}
#endif