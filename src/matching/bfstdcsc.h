/*
 * This source code is distributed under the terms defined  
 * in the file bfstdcsc_main.c of this source distribution.
*/
/* 
*  Breadth first search (BFS) 
*  Single precision (float data type) 
*  TurboBFS_CSC_TD:bfstdcsc.h
* 
*   This program computes: 
*   1) a sequential top-down BFS and
*   2) a single precision GPU-based parallel top-down BFS, for
*      unweighted graphs with sparse matrices in the CSC format.
*      
* 
*/

#ifndef BFSTDCSC_H
#define BFSTDCSC_H

#define MAX_THREADS_PER_GRID (2**31)
#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK/THREADS_PER_WARP)
#define I_SIZE ((3/2)*THREADS_PER_BLOCK)
#include <stdio.h>
#include "CSRGraphRep.h"

/*define Structure of Arrays (SoA) for the sparse matrix A representing
 unweighted graphs in the CSC format*/
struct Csc{
  int   *IC;
  int   *CP;
  
};


typedef struct
{
  /*Allocate device memory for the vector CP_d */
  unsigned int *CP_d;
  unsigned int *IC_d;
  int *degree_d;
  int *base_d;
  int *fll_d;
  int *bll_d;
  int *contraction_counter_d;

  int *blossom_start_d;
  int *blossom_end_d;
  int *vertex_inside_outside_d;

} Graph_GPU;


struct Graph {
  int M,N;
  int vertexNum,edgeNum;

  int i,j;
  int *RP,*JC;
  int *out_degree,*in_degree,*degree;
  int *degreeDist,*degreeDistAc;
  double *degreeDistRel,*degreeDistRelAc;
  int *ICLT,*JCLT,*CPLT,*RPLT;
  struct Csc CscA;
  float *valCLT,*valC;
  int maxDegree,idxMaxDegree;
  double meanDegree,sdDegree;
  int maxInDegree ,idxMaxInDegree,maxOutDegree,idxMaxOutDegree;
  double meanInDegree,sdInDegree,meanOutDegree,sdOutDegree;
  int w;
  int ug;
  int format;
  int p;
  int r;
  int repet;
  int seq;

  struct Csc CscA_hgpu;

  /*Allocate device memory for the vector CP_d */
  int *base_h;
  int *fll_h;
  int *bll_h;

  int *blossom_start_h;
  int *blossom_end_h;
  int *vertex_inside_outside_h;

  int *base_hgpu;
  int *fll_hgpu;
  int *bll_hgpu;
  int *contraction_counter_h;

  Graph_GPU graph_device;

};

typedef struct
{
  /*Allocate device memory for the vector CP_d */
  int *m_d;
  int *req_d;
  int *c;
  int *num_matched_d;
} Match_GPU;


struct match {
    int *m_h, *m_d;
    int *m_hgpu;
    int *req_h, *req_d;
    int *c;
    int *num_matched_h;
    int *matches_h;
    int match_count_h;
    Match_GPU match_device;
};

typedef struct
{
  /*Allocate device memory for the vector CP_d */
  int *S_d;
  int *f_d;
  int *ft_d;
  float *sigma_d, *sigma2_d;
  int *c, *intra_blossom_c;
  int *S_intra_blossom_d;
  int *f_intra_blossom_d;
  int *ft_intra_blossom_d;
  int *pred_d;
  int *forward_d;
  int *search_tree_src_d;
} BFS_GPU;

struct bfs {
    float *sigma, *sigma_hgpu, *sigma_d;
    int *f_h, *ft_h;
    int *f_intra_blossom_h, *ft_intra_blossom_h;
    int *S,*S_hgpu, *S_d;
    int *f_d;
    int *ft_d;
    int *c;
    int *pred_h;
    int *forward_h;
    int *pred_hgpu;
    int *forward_hgpu;
    int *S_intra_blossom_h,*S_intra_blossom_hgpu;

    BFS_GPU bfs_device;
};

typedef struct
{
  /*Allocate device memory for the vector CP_d */
  //int *pred_d;
  // two 32 bit ints packed into 64 bit variable.
  // upper32 lower32
  // dest    src
  // Necessary since all vertices in a blossom write to 1 location.
  // Therefore, the src is unknown.
  int *path_extracted;
  int * pred_d;
  unsigned long long * blossom_start_stop_d;
  int *btype_d;
  int *paths_d;
  int *path_length_d;
  int *S_blossom_d;
  // TODO use width-explicit types "uint64_t"s
  unsigned long long * BTypePair_d;
} Paths_GPU;

struct paths {
    int *pred_h;
    unsigned long long * blossom_start_stop_h;
    int *btype_h;
    int *paths_h;
    int *S_blossom_h;
    int *path_length_h;
    unsigned long long * BTypePair_h;
    Paths_GPU paths_device;
};

typedef struct
{
  /*Allocate device memory for the vector CP_d */
  int *L_d;
  int *c;
} MIS_GPU;

struct MIS
{
  /*Allocate device memory for the vector CP_d */
  int *L_h;
  MIS_GPU mis_device;
};
/**************************************************************************/
/* 
 * Function to compute a sequential top-down BFS for unweighted graphs,
 * represented by sparse adjacency matrices in CSC format, including the  
 * computation of the S vector to store the depth at which each vertex is
 * discovered.    
 *  
*/
int bfs_seq_td_csc (CSRGraph & graph,CSRGraph & graph_d,struct bfs * b);

/**************************************************************************/
/* 
 * Function to compute a sequential top-down m-alternating BFS for unweighted graphs,
 * represented by sparse adjacency matrices in CSC format, including the  
 * computation of the S vector to store the depth at which each vertex is
 * discovered.    
 *  
*/
void bfs_seq_td_csc_m_alt (CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, struct match * m);
void bfs_seq_td_csc_m_alt_w_blossoms (CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, struct match * m);

/**************************************************************************/
/* 
 * Function to compute a gpu-based parallel top-down BFS (scalar) for 
 * unweighted graphs represented by sparse adjacency matrices in CSC format,
 * including the computation of the S vector to store the depth at which each  
 * vertex is  discovered.
 *  
 */
void bfs_gpu_td_csc_sc(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, int exec_protocol);
void bfs_gpu_td_csc_sc_m_alt(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, struct match * mm,int exec_protocol);
void bfs_gpu_td_csc_sc_m_alt_w_blossoms(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, struct match * mm,int exec_protocol);

void simple_graph_builder(FILE *fp, struct Graph* graph, int source_protocol, int exec_protocol);
Graph_GPU create_graph_gpu_struct(CSRGraph & graph);
Match_GPU create_match_gpu_struct(int N);
MIS_GPU create_mis_gpu_struct(int N);
BFS_GPU create_bfs_gpu_struct(int N);
Paths_GPU create_paths_gpu_struct(int N);
void extract_paths_gpu(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b,struct match * mm, struct paths * p,int exec_protocol, int pred_protocol);
void extract_single_path_gpu(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b,struct match * mm, struct paths * p,int exec_protocol);
void extract_single_path_gpu_w_blossoms(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b,struct match * mm, struct paths * p,int exec_protocol);
void process_single_path_gpu_w_blossoms(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b,struct match * mm, struct paths * p,int exec_protocol);
void identify_blossoms_and_launch_bfs_from_inside_vertices(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b,struct match * mm, struct paths * p,int exec_protocol);
int check_graph_gpu(int * CP_d, int * CP_h, int N, int * IC_d, int * IC_h, int edgeNum);

int mm_gpu_csc(CSRGraph & graph,CSRGraph & graph_d,struct match * m, int exec_protocol);
int mm_gpu_csc_from_mis(CSRGraph & graph,CSRGraph & graph_d,struct match * m,struct MIS * mis, int exec_protocol);
int mis_gpu(CSRGraph & graph,CSRGraph & graph_d,struct MIS * mis, int exec_protocol);
//int  mm_gpu_csc (unsigned int *IC_h,unsigned int *CP_h,int *m_h,int *_m_d,int *req_h,int *c_h,int edgeNum,int n,int repetition, int exec_protocol);
void add_edges_to_unmatched_from_last_vertex_cpu_csc(CSRGraph & graph,CSRGraph & graph_d,struct match * m,int exec_protocol);
void add_edges_to_unmatched_from_last_vertex_gpu_csc(CSRGraph & graph,CSRGraph & graph_d,struct match * m,int exec_protocol);
unsigned long mcm_boost(unsigned int *IC_h,unsigned int *CP_h, int N);
unsigned long mcm_boost_headstart(unsigned int *IC_h,unsigned int *CP_h, int *m_h,int N);       // It has a perfect matching of size 8. There are two isolated
unsigned long mcm_edmonds(unsigned int *IC_h,unsigned int *CP_h, int *m_h,int N);


//int edmonds_mcm();
/**************************************************************************/
/* 
 * function to compute a gpu-based parallel top-down BFS (warp shuffle) for
 * unweighted graphs represented by sparse adjacency matrices in CSC format,
 * including the computation of the S vector to store the depth at which each 
 * vertex is discovered.
 *  
 */
void bfs_gpu_td_csc_wa(CSRGraph & graph,CSRGraph & graph_d,struct bfs * b, int exec_protocol);
int  bfs_gpu_td_csc_wa_m_alt  (unsigned int *IC_h,unsigned int *CP_h,int *S_h,float *sigma_h,int *S_hgpu,float *sigma_hgpu,
      int *f_h, int *ft_h, int *c_h, int r,int edgeNum,int n,int repetition,int exec_protocol, int seq);


void set_start_in_blossom_head_seq (int *ft_h,int *base_h,int *blossom_start_h,int n);
void unset_ft_if_not_blossom_start_seq (int *ft_h,int *base_h,int *pred_h,int *blossom_start_h,int n);
/**************************************************************************/
/* 
 * function to compute the sum of two vectors
 * 
*/
int sum_vs (float *sigma,int *f,int n);

/**************************************************************************/
/* 
 * function to check that a vector is 0
 * 
 */
int check_f (int *c,int *f,int n);

void bfsFunctionsKernel_intra_blossom_seq (int *f_d,int *f_intra_blossom_d,int *ft_intra_blossom_d,float *sigma_d,int *S_d,int *S_intra_blossom_d,int *c_intra_blossom,
			 int n,int d,int intra_blossom_d);
void bfsFunctionsKernel_seq (int *f_d,int *ft_d,float *sigma_d,int *S_d,int *c,
			 int n,int d);
/**************************************************************************/
/* 
 * function to compute the S vector to store the depth at which each vertex 
 * is discovered.
 * 
 */
int S_v (int *S,int *f,int n,int d);
int S_v_intra_blossom (int *S,int *S_intra_blossom_h,int *f,int n,int d, int intra_blossom_d);
/**************************************************************************/
/* 
 * function to compute the multiplication of one vector times the complement of the
 * other vector.
*/
int mult_vs (float *sigma,int *f,int n);
/**************************************************************************/
/* 
 * function to assign one vector (different than zero) to another vector. 
 * 
*/
int assign_v(int *f,int *f_t,int n);
int assign_v_intra_blossom(int *f_intra_blossom_h,int *ft_intra_blossom_h,int*f_h,int n);
#endif
