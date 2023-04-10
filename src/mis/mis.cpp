#include "mis.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>
void create_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis, int exec_protocol){
  mis->L_h = (int *) calloc(graph.vertexNum,sizeof(*mis->L_h));
  if(exec_protocol){
    mis->mis_device = create_mis_gpu_struct(graph.vertexNum);
  }
}

void find_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis, int exec_protocol){
    
    mis_gpu (graph,graph_d, mis, exec_protocol);
    if (true){
      check_mis(graph,graph_d, mis);
    }
}

void check_mis(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis){

}