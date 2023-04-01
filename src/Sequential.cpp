#include "Sequential.h"
#include <cstdio>
// DFS from a single source.
unsigned int Sequential(CSRGraph graph, unsigned int minimum)
{
    Stack stack;
    stack.size = graph.vertexNum + 1;

    stack.stack = (int *)malloc(sizeof(int) * stack.size * graph.vertexNum);
    stack.stackNumDeletedVertices = (unsigned int *)malloc(sizeof(int) * stack.size);

    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        stack.stack[j] = graph.degree[j];
    }
    stack.stackNumDeletedVertices[0] = 0;

    stack.top = 0;
    bool popNextItr = true;
    int *vertexDegrees = (int *)malloc(sizeof(int) * graph.vertexNum);
    unsigned int numDeletedVertices;

    while (stack.top != -1)
    {

        if (popNextItr)
        {
            for (unsigned int j = 0; j < graph.vertexNum; ++j)
            {
                vertexDegrees[j] = stack.stack[stack.top * graph.vertexNum + j];
                numDeletedVertices = stack.stackNumDeletedVertices[stack.top];
            }
            --stack.top;
        }

        bool leafHasChanged = false, highDegreeHasChanged = false, triangleHasChanged = false;
        unsigned int iterationCounter = 0;

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

        // Prune branch.  Only way to prune in DFS is if all N(i) are visited.
        if (numDeletedVertices >= minimum || numEdges >= squareSequential(minimum - numDeletedVertices - 1) + 1)
        {
            popNextItr = true;
        }
        else
        {
            // Update best answer before continuing.
            if (maxDegree == 0)
            {
                minimum = numDeletedVertices;
                popNextItr = true;
            }
            // Push degree state onto top and CONTINUE down this branch.
            else
            {
                popNextItr = false;
                ++stack.top;

                for (unsigned int j = 0; j < graph.vertexNum; ++j)
                {
                    stack.stack[stack.top * graph.vertexNum + j] = vertexDegrees[j];
                }
                stack.stackNumDeletedVertices[stack.top] = numDeletedVertices;

                for (unsigned int i = graph.srcPtr[maxVertex]; i < graph.degree[maxVertex] + graph.srcPtr[maxVertex]; ++i)
                {
                    deleteVertex(graph, graph.dst[i], &stack.stack[stack.top * graph.vertexNum], &stack.stackNumDeletedVertices[stack.top]);
                }

                deleteVertex(graph, maxVertex, vertexDegrees, &numDeletedVertices);
            }
        }
    }

    graph.del();
    free(stack.stack);
    free(vertexDegrees);

    return minimum;
}


// DFS from a single source.
// For DFSDFS the stack.top corresponds to the depth.
unsigned int SequentialDFSDFS(CSRGraph graph, unsigned int minimum)
{
    Stack stack;
    stack.foundSolution = false;
    stack.size = graph.vertexNum + 1;
    // Number of DFS levels to do before pushing to the stack.
    //int backtrackingIndices[NUM_LEVELS];
    int * backtrackingIndices =  new int[graph.vertexNum];

    printf("Entered seq\n");
    for (int i = 0; i < graph.vertexNum; i++){
        backtrackingIndices[i]=0;
    }
    unsigned int originalStartVertex;
    stack.stack = (int *) calloc(stack.size * graph.vertexNum,sizeof(int));
    //stack.stack = (int *)malloc(sizeof(int) * stack.size * graph.vertexNum);
    stack.startVertex = (unsigned int *) calloc(stack.size,sizeof(unsigned int));
    //stack.startVertex = (unsigned int *)malloc(sizeof(int) * stack.size);
    stack.top = 0;
    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        stack.stack[j] = 0;
        printf("%d ", j);
    }
    printf("\n");
    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        printf("%d ", graph.matching[j]);
    }
    printf("\n");
    for (unsigned int j = 0; j < graph.num_unmatched_vertices; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        printf("%d ",graph.unmatched_vertices[j]);
    }
    printf("\n");
    stack.startVertex[stack.top] = graph.unmatched_vertices[0];
    stack.stack[stack.startVertex[stack.top]]=1;

    originalStartVertex = graph.unmatched_vertices[0];
    //stack.startVertex[stack.top] = 1;

    bool popNextItr = true;
    //int *visited = (int *)malloc(sizeof(int) * graph.vertexNum);
    int *visited = (int *) calloc(graph.vertexNum,sizeof(int));

    unsigned int startVertex = stack.startVertex[stack.top];
    bool foundSolution = false;
    unsigned int depth;
    while (stack.top != -1 && !foundSolution)
    {
        if (popNextItr)
        {
            printf("Popping %d on the stack top %d\n",stack.startVertex[stack.top],stack.top);
            printf("resetting depth %d backTrackingindices[%d]=%d to 0\n",stack.top+1, stack.top+1,backtrackingIndices[stack.top+1]);
            backtrackingIndices[stack.top+1]=0;
            for (unsigned int j = 0; j < graph.vertexNum; ++j)
            {
                visited[j] = stack.stack[stack.top * graph.vertexNum + j];
            }
            startVertex = stack.startVertex[stack.top];
            --stack.top;
        }
        popNextItr = false;
        bool leafHasChanged = false, highDegreeHasChanged = false, triangleHasChanged = false;
        unsigned int iterationCounter = 0;
        unsigned int neighbor;
        unsigned int start;
        unsigned int end;
        bool foundNeighbor = false;
        printf("popNextItr %d stack.top %d sv %d topsv %d counter %d\n", popNextItr, stack.top, startVertex, stack.startVertex[stack.top],backtrackingIndices[stack.top]);
        printf("depth %d backTrackingindices[%d]=%d sv %d\n",stack.top+1, stack.top+1,backtrackingIndices[stack.top+1],startVertex);

        start = graph.srcPtr[startVertex];
        end = graph.srcPtr[startVertex + 1];
        for (; backtrackingIndices[stack.top+1] < end-start; ++backtrackingIndices[stack.top+1])
        {
            neighbor = graph.dst[start + backtrackingIndices[stack.top+1]];
            printf("%d->%d visited %d\n", startVertex, neighbor, visited[neighbor]);

            if (!visited[neighbor])
            {
                foundNeighbor = true;
                break;
            }
        }
        // Prune branch.  Only way to prune in DFS is if all N(i) are visited.
        if (!foundNeighbor)
        {
            popNextItr = true;
        }
        else
        {
            // Found a solution. Terminate.
            if (graph.matching[neighbor]==-1)
            {
                foundSolution = true;
                printf("Found solution %d\n", neighbor);
            }
            // Push degree state onto top and CONTINUE down this branch.
            //else
            {

                popNextItr = false;
                // Increment starting point of current vertex, 
                // so when it's popped no redundant work is done.
                ++backtrackingIndices[stack.top+1];

                ++stack.top;
                printf("Pushing %d on the stack top %d\n",startVertex,stack.top);

                // Push visited onto stack BEFORE setting neighbor as visited.
                // This way I can pop the visited vertices and take a different route
                // without needing to unset the visited flag of the unsuccessful branch.
                for (unsigned int j = 0; j < graph.vertexNum; ++j)
                {
                    stack.stack[stack.top * graph.vertexNum + j] = visited[j];
                }
                stack.startVertex[stack.top] = startVertex;

                visited[neighbor]=1;
                startVertex = neighbor;
            }
            if (foundSolution){
                ++stack.top;
                stack.startVertex[stack.top] = startVertex;
            }
        }
    }
    printf("Solution:\n");
    for (unsigned int j = 0; j < stack.top+1; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        printf("%d ",stack.startVertex[j]);
    }
    printf("\n");
    graph.del();
    free(stack.stack);
    free(visited);

    return minimum;
}

// DFS from a single source.
unsigned int SequentialBFSDFS(CSRGraph graph, unsigned int minimum)
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
    unsigned int originalStartVertex;

    stack.stack = (int *)malloc(sizeof(int) * stack.size * graph.vertexNum);
    stack.startVertex = (unsigned int *)malloc(sizeof(int) * stack.size);
    stack.top = 0;
    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        stack.stack[j] = 0;
        printf("%d ", j);
    }
    printf("\n");
    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        stack.stack[j] = 0;
        printf("%d ", graph.matching[j]);
    }
    for (unsigned int j = 0; j < graph.num_unmatched_vertices; ++j)
    {
        //stack.stack[j] = graph.visited[j];
        //printf("%d ",graph.unmatched_vertices[j]);
    }
    stack.startVertex[stack.top] = graph.unmatched_vertices[0];
    originalStartVertex = graph.unmatched_vertices[0];
    //stack.startVertex[stack.top] = 1;

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
                // Terminate!!!
                if (graph.matching[path[depth]]==-1 && path[depth] != originalStartVertex){
                    depth = -1;
                    stack.foundSolution = true;
                    printf("Terminate!!!\n");
                    for (unsigned int j = 0; j < graph.vertexNum; ++j)
                    {
                        //stack.stack[j] = graph.visited[j];
                        printf("%d ", j);
                    }
                    printf("\n");
                    for (unsigned int j = 0; j < graph.vertexNum; ++j)
                    {
                        //stack.stack[j] = graph.visited[j];
                        printf("%d ", visited[j]);
                    }
                    break;
                }
                if (graph.matching[path[depth]]>-1 && !visited[graph.matching[path[depth]]]){
                    neighbor = graph.matching[path[depth]];
                    printf("m %d->%d\n",path[depth],neighbor);
                    depth++;
                    path[depth]=neighbor;
                    visited[neighbor]=1;
                    c = 1;
                } else {
                    start = graph.srcPtr[path[depth]];
                    end = graph.srcPtr[path[depth] + 1];
                    for (; backtrackingIndices[depth] < end-start; ++backtrackingIndices[depth])
                    {
                        neighbor = graph.dst[start + backtrackingIndices[depth]];
                        if (!visited[neighbor])
                        {
                            printf("um %d->%d\n",path[depth],neighbor);
                            depth++;
                            path[depth]=neighbor;
                            visited[neighbor]=1;
                            c = 1;
                            break;
                        }
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
            int bt2 = visited[graph.matching[path[depth]]];
            visited[path[depth]]=0;
            path[depth]=0;
            depth--;
            if(bt2 && depth >= 0){
                path[depth]=0;
                depth--;
            }
            ++backtrackingIndices[depth];    
            c = 1;
        }
        popNextItr = !stack.foundSolution;
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