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

#include <queue>
#include <bits/stdc++.h>
#include "edmonds.cuh"
#include "bfstdcsc.h"

//using namespace std;


Edmonds::Edmonds(int *IC_h,int *CP_h, int *m_h, int N):
  IC(IC_h),CP(CP_h),match(m_h), V(N)
  {
  //match = (int *)malloc(V * sizeof(int));
  father = (int *)malloc(V * sizeof(int));
  base = (int *)malloc(V * sizeof(int));
  
  q = (int *)malloc(V * sizeof(int));
  inq = (bool *)malloc(V * sizeof(bool));
  inb = (bool *)malloc(V * sizeof(bool));
  inp = (bool *)malloc(V * sizeof(bool));

}

int Edmonds::LCA(int root,int u,int v)
{
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
void Edmonds::mark_blossom(int lca,int u)
{
  while (base[u]!=lca)
    {
      int v=match[u];
      inb[base[u]]=inb[base[v]]=true;
      u=father[v];
      if (base[u]!=lca) father[u]=v;
    }
}
void Edmonds::blossom_contraction(int s,int u,int v)
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
int Edmonds::find_augmenting_path(int s)
{
  memset(inq, 0, sizeof(bool) * V);
  memset(father, -1, sizeof(int) * V);

  for (int i=0;i<V;i++) base[i]=i;
  inq[q[qh=qt=0]=s]=true;
  while (qh<=qt)
    {
      int u=q[qh++];
      int start = CP[u];
      int end = CP[u+1];
      int k;
      for (k=start; k<end; k++){
        int v=IC[k];
        if (base[u]!=base[v]&&match[u]!=v)
          if ((v==s)||(match[v]!=-1 && father[match[v]]!=-1))
            blossom_contraction(s,u,v);
          else if (father[v]==-1)
          {
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
int Edmonds::augment_path(int s,int t)
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
int Edmonds::max_cardinality_matching()
{
  int matchc=0;
  //memset(match, -1, sizeof(int) * V);

  for (int u=0;u<V;u++)
    if (match[u]==-1)
      matchc+=augment_path(u,find_augmenting_path(u));
  return matchc;
}

unsigned long mcm_edmonds(int *IC_h,int *CP_h,int *m_h,int N)
{
  Edmonds ed(IC_h,CP_h,m_h,N);
  return ed.max_cardinality_matching();
}
