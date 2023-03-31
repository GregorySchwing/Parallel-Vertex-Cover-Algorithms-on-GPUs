#include "Sequential.h"
#include <cstdio>
// DFS from a single source.
unsigned int Sequential(CSRGraph graph, unsigned int minimum)
{
    Stack stack;
    stack.foundSolution = false;
    stack.size = graph.vertexNum + 1;
    // Number of DFS levels to do before pushing to the stack.
    int backtrackingIndices[NUM_LEVELS];
    int path[NUM_LEVELS+ 1];
    printf("Entered seq\n");
    for (int i = 0; i < NUM_LEVELS; i++){
        backtrackingIndices[i]=0;
        path[i]=0;
    }

    stack.stack = (int *)malloc(sizeof(int) * stack.size * graph.vertexNum);
    stack.startVertex = (unsigned int *)malloc(sizeof(int) * stack.size);
    stack.top = 0;

    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        stack.stack[j] = 0;
    }
    stack.startVertex[stack.top] = 1;

    bool popNextItr = true;
    int *visited = (int *)malloc(sizeof(int) * graph.vertexNum);
    unsigned int startVertex;

    while (stack.top != -1 && !stack.foundSolution)
    {

        if (popNextItr)
        {
            for (unsigned int j = 0; j < graph.vertexNum; ++j)
            {
                visited[j] = stack.stack[stack.top * graph.vertexNum + j];
            }
            startVertex = stack.startVertex[stack.top];
            printf("Popping %d on the stack top %d\n",startVertex,stack.top);
            for (int i = 0; i < NUM_LEVELS; i++){
                backtrackingIndices[i]=0;
                path[i]=0;
            }
            --stack.top;
        }
        popNextItr = false;
        bool leafHasChanged = false, highDegreeHasChanged = false, triangleHasChanged = false;
        unsigned int iterationCounter = 0;

        // Do the DFS work.
        // 1 level
        int depth = 0;
        path[depth]=startVertex;
        visited[startVertex] = 1;
        int c = 1;
        unsigned int neighbor;
        unsigned int start;
        unsigned int end;
        while(depth > -1){
            while(c && depth < NUM_LEVELS){
                //for (depth = 0; depth < NUM_LEVELS; ++depth){
                c = 0;
                printf("Depth %d\n",depth);
                start = graph.srcPtr[path[depth]];
                end = graph.srcPtr[path[depth] + 1];
                for (; backtrackingIndices[depth] < end-start; ++backtrackingIndices[depth])
                {
                    neighbor = graph.dst[start + backtrackingIndices[depth]];
                    if (!visited[neighbor])
                    {
                        printf("%d->%d\n",path[depth],neighbor);
                        depth++;
                        path[depth]=neighbor;
                        visited[neighbor]=1;
                        c = 1;
                        break;
                    }
                }
            }
            // Push end of local search tree to stack
            // c must be true
            if (depth==NUM_LEVELS){        
                ++stack.top;
                for (unsigned int j = 0; j < graph.vertexNum; ++j)
                {
                    stack.stack[stack.top * graph.vertexNum + j] = visited[j];
                }
                stack.startVertex[stack.top] = neighbor;
                printf("Pushing %d on the stack top %d\n",neighbor,stack.top);
            } 
            printf("Depth %d\n",depth);
            visited[path[depth]]=0;
            path[depth]=0;
            depth--;
            ++backtrackingIndices[depth];    
            c=1;        
        }
        popNextItr = true;
        //}
        /*
        do
        {
            leafHasChanged = leafReductionRule(graph, vertexDegrees, &numDeletedVertices);
            triangleHasChanged = triangleReductionRule(graph, vertexDegrees, &numDeletedVertices);
            highDegreeHasChanged = highDegreeReductionRule(graph, vertexDegrees, &numDeletedVertices, minimum);

        } while (triangleHasChanged || highDegreeHasChanged);

        unsigned int maxVertex = 0;
        int maxDegree = 0;
        unsigned int numEdges = 0;

        for (unsigned int i = 0; i < graph.vertexNum; ++i)
        {
            int degree = vertexDegrees[i];
            if (degree > maxDegree)
            {
                maxDegree = degree;
                maxVertex = i;
            }

            if (degree > 0)
            {
                numEdges += degree;
            }
        }

        numEdges /= 2;
        */

        /*

        if (numDeletedVertices >= minimum || numEdges >= squareSequential(minimum - numDeletedVertices - 1) + 1)
        {
            popNextItr = true;
        }
        else
        {
            if (maxDegree == 0)
            {
                minimum = numDeletedVertices;
                popNextItr = true;
            }
            else
            {
                popNextItr = false;
                ++stack.top;

                for (unsigned int j = 0; j < graph.vertexNum; ++j)
                {
                    stack.stack[stack.top * graph.vertexNum + j] = vertexDegrees[j];
                }
                stack.startVertex[stack.top] = numDeletedVertices;

                for (unsigned int i = graph.srcPtr[maxVertex]; i < graph.degree[maxVertex] + graph.srcPtr[maxVertex]; ++i)
                {
                    deleteVertex(graph, graph.dst[i], &stack.stack[stack.top * graph.vertexNum], &stack.startVertex[stack.top]);
                }

                deleteVertex(graph, maxVertex, vertexDegrees, &numDeletedVertices);
            }
        }
        */
    }

    graph.del();
    free(stack.stack);
    free(visited);

    return minimum;
}