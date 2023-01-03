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
    int FindXq(int u, int w);

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
              printf("dist from %d to %d : %d\n", u+1, w+1, dist[w]);
            } else if(dist[w] == depth){
              printf("M alternating Cycle detected %d %d at depth %d!!!\n", u, w, depth);
              int xq = FindXq(u,w);
              printf("xq %d\n", xq+1);
              exit(1);
            }
          } else {
            if(dist[w] == -1 && match[u] == w && match[w] == u){
              dist[w] = dist[u] + 1;
              pred[w] = u;
              printf("dist from %d to %d : %d\n", u+1, w+1, dist[w]);
            } else if(dist[w] == depth && match[u] == w && match[w] == u){
              printf("M alternating Cycle detected %d %d at depth %d!!!\n", u, w, depth);
              int xq = FindXq(u,w);
              printf("xq %d\n", xq+1);
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

int CrownDecomposition::FindXq(int u, int w){
  // Simply grab parents. necessarily distinct.
  if (dist[u] % 2 == 0){
    printf("%d -> %d\n", u,pred[u]);
    printf("%d -> %d\n", w, pred[w]);
    u = pred[u];
    w = pred[w];
    return FindXq(u, w);
  } else {
    // I'm looking for a v ∈ Neighborhood(u), s.t. dist(start,v) < dist(start,u)
    // where v ≠ pred[w]
    int distU = dist[u];
    int predW = pred[w];
    for (unsigned int j = graph.srcPtr[u]; j < graph.srcPtr[u + 1]; ++j){
      unsigned int v =graph.dst[j];
      printf("Neighbor of %d : %d (%d) \n", u+1, v+1, dist[v]);
      if (v != predW && dist[v] != -1 && dist[v] < distU){
        printf("%d -> %d\n", u,v);
        printf("%d -> %d\n", w, predW);
        u = v;
        w = predW;
        return FindXq(u, w);
      }
    }      
    int distW = dist[w];
    int predU = pred[u];
    for (unsigned int j = graph.srcPtr[w]; j < graph.srcPtr[w + 1]; ++j){
      unsigned int v =graph.dst[j];
      printf("Neighbor of %d : %d (%d) \n", w+1, v+1, dist[v]);
      if (v != predU && dist[v] != -1 && dist[v] < distW){
        printf("%d -> %d\n", u,predU);
        printf("%d -> %d\n", w, v);
        u = predU;
        w = v;
        return FindXq(u, w);
      }
    }
    return predU;
  }
}
#endif