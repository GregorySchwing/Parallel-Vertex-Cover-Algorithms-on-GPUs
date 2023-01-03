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
        dist = (int *)malloc(V * sizeof(int));
        pred = (int *)malloc(V * sizeof(int));
        int mc = mmb.edmonds();
        printf("Match count %d\n", 2*mc);
        match = mmb.get_match();
    }
    ~CrownDecomposition(){

        free(dist);
        free(pred);
    }

    bool FindCrown();
    private:
        CSRGraph &graph;
        MaximumMatcherBlossom mmb;
        // crown variables
        int *dist, *pred, *match;
        int V,E;

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
#endif