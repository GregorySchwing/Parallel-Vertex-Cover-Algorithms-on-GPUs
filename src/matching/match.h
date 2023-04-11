

/*
 *  * Functions to:
 *   1) compute the parameters(maximum, mean, and standard
 *      deviation) of the degree, out degree and in degree
 *      vectors
 *   2) compute the degree vector for undirected graphs.
 *
 */

#ifndef MATCH_H
#define MATCH_H

//#include "graph.h"
//#include "bfs.h"
#include "CSRGraphRep.h"
#include "ThrustGraph.h"

#include "mis.h"
#include "bfstdcsc.h"
void create_match(int vertexNum,struct match * m, int exec_protocol);
void create_match(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol);
void maxmatch(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol);
void maxmatch_thrust(ThrustGraph & graph, struct match * m, int exec_protocol);
void maxmatch_from_mis(CSRGraph & graph,CSRGraph & graph_d,struct match * m, struct MIS * mis, int exec_protocol);
void add_edges_to_unmatched_from_last_vertex(ThrustGraph & graph,struct match * m,int exec_protocol);
void add_edges_to_unmatched_from_last_vertex(CSRGraph & graph,CSRGraph & graph_d,struct match * m,int exec_protocol);
void set_unmatched_as_source(unsigned int *IC, unsigned int *CP, int * m_h, int n, int edgeNum);
void check_match(CSRGraph & graph,CSRGraph & graph_d,struct match * m);
void check_IC_CP(unsigned int *IC, unsigned int *CP, unsigned int *IC_hgpu, unsigned int *CP_hgpu, int n_ub, int nz_ub);
#endif