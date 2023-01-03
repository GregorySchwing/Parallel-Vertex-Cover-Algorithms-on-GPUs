#ifndef MAXIMUM_MATCHING_H
#define MAXIMUM_MATCHING_H
#include <limits>
#include <bits/stdc++.h>
using namespace std;

/*

https://codeforces.com/blog/entry/49402

GETS:
V->number of vertices
E->number of edges
pair of vertices as edges (vertices are 1..V)

GIVES:
output of edmonds() is the maximum matching
match[i] is matched pair of i (-1 if there isn't a matched pair)
 */
struct struct_edge{int v;struct_edge* n;};
typedef struct_edge* edge;

 
class MaximumMatcherBlossom {
  public:
    MaximumMatcherBlossom(CSRGraph &_G) : 
                    graph(_G),
                    V(_G.vertexNum),
                    E(_G.edgeNum)
    {
      match = (int *)malloc(V * sizeof(int));
      q = (int *)malloc(V * sizeof(int));
      father = (int *)malloc(V * sizeof(int));
      base = (int *)malloc(V * sizeof(int));

      inq = (bool *)malloc(V * sizeof(bool));
      inb = (bool *)malloc(V * sizeof(bool));
    }
    ~MaximumMatcherBlossom(){
      free(match);
      free(q);
      free(father);
      free(base);

      free(inq);
      free(inb);
    }

    int* get_match(){return match;}

    int edmonds();

    private:
        CSRGraph &graph;
        int V,E,*match,qh,qt,*q,*father,*base;
        bool *inq,*inb;
        //struct_edge pool[M*M*2];
        //edge top=pool,adj[M];
        //int V,E,match[M],qh,qt,q[M],father[M],base[M];
        //bool inq[M],inb[M],ed[M][M];

        void add_edge(int u,int v);
        int LCA(int root,int u,int v);
        void mark_blossom(int lca,int u);
        void blossom_contraction(int s,int u,int v);
        int find_augmenting_path(int s);
        int augment_path(int s,int t);

};

int MaximumMatcherBlossom::LCA(int root,int u,int v)
{
  static bool * inp = (bool *)malloc(V * sizeof(bool));
  memset(inp, 0, sizeof(bool) * V);
  while(1)
    {
      inp[u=base[u]]=true;
      if (u==root) break;
      u=father[match[u]];
    }
  while(1)
    {
      if (inp[v=base[v]]) return v;
      else v=father[match[v]];
    }
    free(inp);
}
void MaximumMatcherBlossom::mark_blossom(int lca,int u)
{
  while (base[u]!=lca)
    {
      int v=match[u];
      inb[base[u]]=inb[base[v]]=true;
      u=father[v];
      if (base[u]!=lca) father[u]=v;
    }
}
void MaximumMatcherBlossom::blossom_contraction(int s,int u,int v)
{
  int lca=LCA(s,u,v);
  memset(inb, 0, sizeof(bool) * V);

  mark_blossom(lca,u);
  mark_blossom(lca,v);
  if (base[u]!=lca)
    father[u]=v;
  if (base[v]!=lca)
    father[v]=u;
  for (int u=0;u<V;u++)
    if (inb[base[u]])
      {
	base[u]=lca;
	if (!inq[u])
	  inq[q[++qt]=u]=true;
      }
}
int MaximumMatcherBlossom::find_augmenting_path(int s)
{
  memset(inq, 0, sizeof(bool) * V);
  memset(father, -1, sizeof(int) * V);

  for (int i=0;i<V;i++) base[i]=i;
  inq[q[qh=qt=0]=s]=true;
  while (qh<=qt)
    {
      int u=q[qh++];
      if (graph.degree[u] <= 0)
        continue;
      for (unsigned int j = graph.srcPtr[u]; j < graph.srcPtr[u + 1]; ++j){
        unsigned int v =graph.dst[j];
        if (graph.degree[v] <= 0)
          continue;
        if (base[u]!=base[v]&&match[u]!=v)
          if ((v==s)||(match[v]!=-1 && father[match[v]]!=-1))
            blossom_contraction(s,u,v);
          else if (father[v]==-1){
            father[v]=u;
            if (match[v]==-1)
              return v;
            else if (!inq[match[v]])
              inq[q[++qt]=match[v]]=true;
          }
        }
    }
  return -1;
}
int MaximumMatcherBlossom::augment_path(int s,int t)
{
  int u=t,v,w;
  while (u!=-1)
    {
      v=father[u];
      w=match[v];
      match[v]=u;
      match[u]=v;
      u=w;
    }
  return t!=-1;
}
int MaximumMatcherBlossom::edmonds()
{
  int matchc=0;
  memset(match, -1, sizeof(int) * V);

  for (int u=0;u<V;u++)
    if (match[u]==-1)
      matchc+=augment_path(u,find_augmenting_path(u));
  return matchc;
}
#endif