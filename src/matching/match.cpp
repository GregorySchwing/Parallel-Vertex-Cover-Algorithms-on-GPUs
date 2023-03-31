#include "match.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>
void create_match(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol){
  m->m_h = (int *) calloc(graph.vertexNum,sizeof(*m->m_h));
  m->m_hgpu = (int *) calloc(graph.vertexNum,sizeof(*m->m_h));
  m->num_matched_h = (int *) calloc(1,sizeof(*m->num_matched_h));
  m->matches_h = (int *) calloc(graph.vertexNum,sizeof(*m->m_h));

  if(exec_protocol){
    m->match_device = create_match_gpu_struct(graph.vertexNum);
  }
}

void maxmatch(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol){
    mm_gpu_csc (graph,graph_d, m, exec_protocol);
    if (true){
      check_match(graph,graph_d,m);
    }
}


void maxmatch_from_mis(CSRGraph & graph,CSRGraph & graph_d,struct match * m, struct MIS* mis, int exec_protocol){
    mm_gpu_csc_from_mis (graph,graph_d,m, mis,exec_protocol);
    if (true){
      check_match(graph,graph_d,m);
    }
}

void add_edges_to_unmatched_from_last_vertex(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol){
  add_edges_to_unmatched_from_last_vertex_gpu_csc (graph,graph_d,m,exec_protocol);
  //if (true)
  //  add_edges_to_unmatched_from_last_vertex_cpu_csc (graph,graph_d,m,exec_protocol);

}

void check_match(CSRGraph & graph,CSRGraph & graph_d,struct match * m){
  unsigned int *IC = graph.dst;
  unsigned int *CP = graph.srcPtr;
  int start, end, k, counter = 0, found = 1, matchViolations = 0;
  memset(m->matches_h, 0, graph.vertexNum*sizeof(*m->matches_h));
  m->match_count_h = 0;
  for (int i = 0; i < graph.vertexNum; ++i){
    found = 1;
    //printf("%d %d\n",i,m->m_h[i]);
    if (m->m_h[i]>-1){
      m->match_count_h +=1;
      start = CP[i];
      end = CP[i+1];
      for (k=start; k<end; k++){
        if (IC[k] == m->m_h[i]){
          //if (print_t)
            //printf("%d -> %d found in graph!\n", i, m->m_h[i]);
          found = 0;
        }
      }
      int print_t = 1;
      if (print_t && found){
        printf("%d -> %d not found in graph!\n", i, m->m_h[i]);
      }
      m->matches_h[m->m_h[i]]+=1;
      counter += found;
    }
  }

  for (int i = 0; i < graph.vertexNum; ++i){
    if (m->matches_h[m->m_h[i]] > 1){
      printf("Match violation v %d u %d c %d\n",i,m->m_h[i],m->matches_h[m->m_h[i]]);
      matchViolations += 1;
    }
  }

  if (counter){
    printf("%d errors in matching!\n", counter);
    exit(1);
  }else 
    printf("Matching is correct.\n");

  if (matchViolations){
    printf("%d violations in matching!\n", matchViolations);
    exit(1);
  }else 
    printf("Matching is correct.\n");
  printf("Matched vertices %d; matched edges %d\n", m->match_count_h, m->match_count_h/2);
}
/* 
 * Function to compute a gpu-based parallel maximal matching for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format.
 *  
 */
void add_edges_to_unmatched_from_last_vertex_cpu_csc(CSRGraph & graph,CSRGraph & graph_d,struct match * m,int exec_protocol){

    unsigned int *IC = graph.dst;
    unsigned int *CP = graph.srcPtr;
    int * m_h = m->m_h;
    unsigned int * IC_hgpu = graph.dst;
    unsigned int * CP_hgpu = graph.srcPtr;

    /* timing variables  */
    double initial_t;
    double delta;
    double total_t = 0.0;
    
    //initial_t = get_time();
    set_unmatched_as_source (IC,CP,m_h,graph.vertexNum,graph.edgeNum);
    //delta = get_time()-initial_t;
    total_t += delta;

    check_IC_CP (IC,CP,IC_hgpu,CP_hgpu,graph.vertexNum+1,graph.edgeNum*2);

    //printf("\nbfs_seq_ug_csc_m_alt::d = %d,r = %d \n",d,r);

    
    return;
}//end bfs_gpu_td_csc_sc

void set_unmatched_as_source(unsigned int *IC, unsigned int *CP, int * m_h, int n, int edgeNum){
  int counter = 0;
  for (int i = 0; i < n; ++i){
    //printf("%d -> %d\n", i, m_h[i]);
    if (m_h[i]<0){
      IC[edgeNum+counter]=i;
      counter++;
    }
  }
  CP[n+1]=edgeNum+counter;
}

void check_IC_CP(unsigned int *IC, unsigned int *CP, unsigned int *IC_hgpu, unsigned int *CP_hgpu, int n_ub, int nz_ub){
  int counter = 0;
  for (int i = 0; i < n_ub+1; ++i){
    if (CP[i]!=CP_hgpu[i]){
      printf("Error CP[%d]=%d CP_hgpu[%d]=%d\n", i,CP[i],i,CP_hgpu[i]);
      counter++;
    }
  }
  if(counter){
    printf("error in %d values of the GPU computed CP_hgpu vector\n",counter);
  }else{
    printf("values of the GPU computed CP vector are correct \n");
  }	
  counter = 0;
  for (int i = 0; i < nz_ub; ++i){
    if (IC[i]!=IC_hgpu[i]){
      counter++;
    }
  }
  if(counter){
    printf("error in %d values of the GPU computed IC_hgpu vector\n",counter);
  }else{
    printf("values of the GPU computed IC vector are correct \n");
  }	
}