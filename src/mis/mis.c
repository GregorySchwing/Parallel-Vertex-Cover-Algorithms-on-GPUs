#include "mis.h"

void create_mis(struct Graph * graph,struct MIS * mis, int exec_protocol){
  mis->L_h = (int *) calloc(graph->N,sizeof(*mis->L_h));
  if(exec_protocol){
    mis->mis_device = create_mis_gpu_struct(graph->N);
  }
}

void find_mis(struct Graph * graph,struct MIS * mis, int exec_protocol){
    
    mis_gpu (graph, mis, exec_protocol);
    if (graph->seq){
      check_mis(graph, mis);
    }
}

void check_mis(struct Graph * graph,struct MIS * mis){

}