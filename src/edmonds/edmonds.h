/*
 *  * Functions to:
 *   1) compute serial mcm using boost
 *
 */

#ifndef EDMONDSMCM_H
#define EDMONDSMCM_H

#include "graph.h"
#include "match.h"

unsigned long create_mcm_edmonds(struct Graph * graph,struct match * mm);

#endif