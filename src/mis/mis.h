

/*
 *  * Functions to:
 *   1) compute the parameters(maximum, mean, and standard
 *      deviation) of the degree, out degree and in degree
 *      vectors
 *   2) compute the degree vector for undirected graphs.
 *
 */

#ifndef MIS_H
#define MIS_H

#include "graph.h"

void create_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis,int exec_protocol);
void find_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis, int exec_protocol);
void check_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis);
#endif