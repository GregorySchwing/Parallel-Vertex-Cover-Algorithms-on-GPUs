

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

#include "graph.h"
#include "bfs.h"
#include "mis.h"

void create_match(struct Graph * graph,struct match * m, int exec_protocol);
void maxmatch(struct Graph * graph,struct match * m, int exec_protocol);
void maxmatch_from_mis(struct Graph * graph,struct match * m, struct MIS * mis, int exec_protocol);
void add_edges_to_unmatched_from_last_vertex(struct Graph * graph,struct match * m,int exec_protocol);
void set_unmatched_as_source(int *IC, int *CP, int * m_h, int n, int nz);
void check_match(struct Graph * graph,struct match * m);
void check_IC_CP(int *IC, int *CP, int *IC_hgpu, int *CP_hgpu, int n_ub, int nz_ub);
#endif