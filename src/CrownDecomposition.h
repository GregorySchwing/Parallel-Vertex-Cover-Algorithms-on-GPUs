#ifndef CROWN_DECOMPOSITION_H
#define CROWN_DECOMPOSITION_H
#include "MaximumMatching.h"

#include <limits>
#include <bits/stdc++.h>
using namespace std;
 
class CrownDecomposition {
  public:
    CrownDecomposition(CSRGraph &_G) : 
                    graph(_G),
                    V(_G.vertexNum),
                    E(_G.edgeNum),
                    mmb(_G)
    {
        
        // BFS variables
        frontier = (bool *)malloc(V * sizeof(bool));
        dist = (int *)malloc(V * sizeof(int));
        pred = (int *)malloc(V * sizeof(int));
        int mc = mmb.edmonds();
        printf("Match count %d\n", 2*mc);
        match = mmb.get_match();
    }
    ~CrownDecomposition(){
        free(frontier);
        free(dist);
        free(pred);
    }

    bool FindCrown();
    bool FindCrownPar(unsigned int &minimum);
    private:
        CSRGraph &graph;
        MaximumMatcherBlossom mmb;
        // crown variables
        int *dist, *pred, *match;
        int V,E;
        bool *frontier;

};

bool CrownDecomposition::FindCrown()
{
  bool foundCrown = false;
  memset(dist, -1, sizeof(int) * V);
  memset(pred, -1, sizeof(int) * V);
  int start = 0;
  for (int i=0;i<V;i++)
    if (match[i] == -1 && graph.degree[i] > 0){
      start = i;
      break;
    }
  dist[start] = 0;
  std::queue<int> queue;
  queue.push(start);
  while(queue.size()){
    int u = queue.front();
    queue.pop();
    for (unsigned int j = graph.srcPtr[u]; j < graph.srcPtr[u + 1]; ++j){
      unsigned int w =graph.dst[j];
      if (dist[u] % 2 == 0){
        if(dist[w] == -1){
          dist[w] = dist[u] + 1;
          pred[w] = u;
          queue.push(w);
        }
      } else {
        if(dist[w] == -1 && match[u] == w && match[w] == u){
          dist[w] = dist[u] + 1;
          pred[w] = u;
          queue.push(w);
        }
      }
    }
  }
  printf("Finished BFS\n");
  for (int i=0;i<V;i++){
    if (dist[i] != -1)
    printf("dist from start %d to %d : %d\n", start+1, i+1, dist[i]);
  }
  return foundCrown;
}

bool CrownDecomposition::FindCrownPar(unsigned int &minimum)
{
  bool foundCrown = false;
  int depth = 0;
  int frontierSum = 1;
  memset(dist, -1, sizeof(int) * V);
  memset(pred, -1, sizeof(int) * V);
  memset(frontier, false, sizeof(bool) * V);
  int start = 0;
  for (int i=0;i<V;i++)
    if (match[i] == -1 && graph.degree[i] > 0){
      start = i;
      break;
    }
  dist[start] = 0;
  frontier[start] = true;
  while(frontierSum){
    for (unsigned int u = 0; u < graph.vertexNum; u++){
      if (frontier[u]){
        for (unsigned int j = graph.srcPtr[u]; j < graph.srcPtr[u + 1]; ++j){
          unsigned int w =graph.dst[j];
          if (dist[u] % 2 == 0){
            if(dist[w] == -1){
              dist[w] = dist[u] + 1;
              pred[w] = u;
            } else if(dist[w] == depth){
              printf("M alternating Cycle detected %d %d at depth %d!!!\n", u, w, depth);
              exit(1);
            }
          } else {
            if(dist[w] == -1 && match[u] == w && match[w] == u){
              dist[w] = dist[u] + 1;
              pred[w] = u;
            } else if(dist[w] == depth && match[u] == w && match[w] == u){
              printf("M alternating Cycle detected %d %d at depth %d!!!\n", u, w, depth);
              exit(1);
            }
          }
        }
      }
    }
    ++depth;
    frontierSum = 0;
    for (unsigned int i = 0; i < graph.vertexNum; i++){
      frontier[i] = dist[i] == depth;
      frontierSum += frontier[i];
    }
  }
  printf("Finished BFS\n");
  for (int i=0;i<V;i++){
    if (dist[i] != -1){
      if(dist[i] % 2 == 0){
        // I set. dont add to soln
      } else {
        // H set. add to soln
        printf("vertex %d is in Head. Add to solution.\n", i+1);
        graph.deleteVertex(i);
        ++minimum;
        foundCrown = true;
      }
      printf("dist from start %d to %d : %d\n", start+1, i+1, dist[i]);
    }
  }
  return foundCrown;
}
#endif