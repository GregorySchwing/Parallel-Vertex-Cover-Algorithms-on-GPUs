#ifndef REDUCTION_RULES_H
#define REDUCTION_RULES_H

#include "Graph.h"
#include "config.h"

template <class Graph>
class ReductionRules {
public:
    static unsigned int RemoveMaxApproximateMVC(const Graph & graph);
    static unsigned int RemoveEdgeApproximateMVC(const Graph & graph);
private:
    static bool auxBinarySearch(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x);
    static bool leafReductionRule(Graph &graph, unsigned int &minimum);
    static bool triangleReductionRule(Graph &graph, unsigned int &minimum);
    static unsigned int getRandom(int lower, int upper);
};

template <class Graph>
unsigned int ReductionRules<Graph>::RemoveMaxApproximateMVC(const Graph & graph){

	Graph approxGraph = graph;

	unsigned int minimum = 0;
	bool hasEdges = true;
	while (hasEdges)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = leafReductionRule(approxGraph, minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = triangleReductionRule(approxGraph, minimum);
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

	//approxGraph.del();

	return minimum;
}

template <class Graph>
unsigned int ReductionRules<Graph>::RemoveEdgeApproximateMVC(const Graph & graph){

	Graph approxGraph = graph;


	unsigned int minimum = 0;
	unsigned int numRemainingEdges = approxGraph.edgeNum;

	for (unsigned int vertex = 0; vertex < approxGraph.vertexNum && numRemainingEdges > 0; vertex++)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = leafReductionRule(approxGraph, minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = triangleReductionRule(approxGraph, minimum);
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

template <class Graph>
bool ReductionRules<Graph>::leafReductionRule(Graph &graph, unsigned int &minimum){
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < graph.vertexNum; ++i)
		{
			if (graph.degree[i] == 1)
			{
				hasChanged = true;
				for (unsigned int j = graph.srcPtr[i]; j < graph.srcPtr[i + 1]; ++j)
				{
					if (graph.degree[graph.dst[j]] != -1)
					{
						unsigned int neighbor = graph.dst[j];
						graph.deleteVertex(neighbor);
						++minimum;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

template <class Graph>
bool ReductionRules<Graph>::triangleReductionRule(Graph &graph, unsigned int &minimum){
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < graph.vertexNum; ++i)
		{
			if (graph.degree[i] == 2)
			{

				unsigned int neighbor1, neighbor2;
				bool foundNeighbor1 = false, keepNeighbors = false;
				for (unsigned int edge = graph.srcPtr[i]; edge < graph.srcPtr[i + 1]; ++edge)
				{
					unsigned int neighbor = graph.dst[edge];
					int neighborDegree = graph.degree[neighbor];
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
					bool found = auxBinarySearch(graph.dst, graph.srcPtr[neighbor2], graph.srcPtr[neighbor2 + 1] - 1, neighbor1);

					if (found)
					{
						hasChanged = true;
						// Triangle Found
						graph.deleteVertex(neighbor1);
						graph.deleteVertex(neighbor2);
						minimum += 2;
						break;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

template <class Graph>
bool ReductionRules<Graph>::auxBinarySearch(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x)
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

template <class Graph>
unsigned int  ReductionRules<Graph>::getRandom(int lower, int upper)
{
	srand(time(0));
	unsigned int num = (rand() % (upper - lower + 1)) + lower;
	return num;
}

#endif
