/*
PCSR - Adapted from Wheaton's PCSR - https://github.com/wheatman/Packed-Compressed-Sparse-Row
MIT License

Copyright (c) 2018 wheatman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef PCSR_GRAPH_POLICY_H
#define PCSR_GRAPH_POLICY_H


#include <algorithm>
#include <queue>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tuple>

using namespace std;

typedef struct _node {
  // beginning and end of the associated region in the edge list
  uint32_t beginning;     // deleted = max int
  uint32_t end;           // end pointer is exclusive
  uint32_t num_neighbors; // number of edges with this node as source
} node_t;

// each node has an associated sentinel (max_int, offset) that gets back to its
// offset into the node array
// UINT32_MAX
//
// if value == UINT32_MAX, read it as null.
typedef struct _edge {
  uint32_t dest; // destination of this edge in the graph, MAX_INT if this is a
                 // sentinel
  uint32_t
      value; // edge value of zero means it a null since we don't store 0 edges
} edge_t;

typedef struct edge_list {
  int N;
  int H;
  int logN;
  edge_t *items;
} edge_list_t;

// find index of first 1-bit (least significant bit)
static inline int bsf_word(int word) {
  int result;
  __asm__ volatile("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline int bsr_word(int word) {
  int result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

typedef struct _pair_int {
  int x; // length in array
  int y; // depth
} pair_int;

typedef struct _pair_double {
  double x;
  double y;
} pair_double;

int isPowerOfTwo(int x) { return ((x != 0) && !(x & (x - 1))); }

// same as find_leaf, but does it for any level in the tree
// index: index in array
// len: length of sub-level.
int find_node(int index, int len) { return (index / len) * len; }

template <class T>
struct PCSR
{
    public:
        PCSR(FILE * fp, u_int32_t  vertexNum, u_int32_t  edgeNum);
        
        ~PCSR();

        void clear();
        vector<float> pagerank(std::vector<float> const &node_values);
        uint64_t get_n();
        bool is_null(edge_t e);
        vector<tuple<uint32_t, uint32_t, uint32_t>> get_edges();
        vector<uint32_t> get_degrees();
        vector<uint32_t> bfs(uint32_t start_node);
        uint64_t get_size();
        void print_array();
        bool is_sentinel(edge_t e);
        void fix_sentinel(int32_t node_index, int in);
        void redistribute(int index, int len);                                                   
        void double_list();
        void half_list();
        int slide_right(int index);
        void slide_left(int index);
        uint32_t find_value(uint32_t src, uint32_t dest);
        uint32_t insert(uint32_t index, edge_t elem, uint32_t src);
        uint32_t binary_search(edge_list_t *list, edge_t *elem, uint32_t start, uint32_t end); 
        int find_leaf(edge_list_t *list, int index);
        double get_density(edge_list_t *list, int index, int len);
        pair_double density_bound(edge_list_t *list, int depth);
        uint32_t find_elem_pointer(edge_list_t *list, uint32_t index, edge_t elem);
        bool edge_equals(edge_t e1, edge_t e2);
        std::vector<uint32_t> sparse_matrix_vector_multiplication(std::vector<uint32_t> const &v);
        void print_graph();
        void printGraph();
        void add_node();
        void add_edge(uint32_t src, uint32_t dest, uint32_t value);
        void add_edge_update(uint32_t src, uint32_t dest, uint32_t value);
        // data members
        std::vector<node_t> nodes;
        edge_list_t edges;
};

template <class T>
PCSR<T>::PCSR(FILE * fp, u_int32_t vertexNum, u_int32_t edgeNum)
{
    printf("\nConstructed PCSR-based Graph\n");

    edges.N = 2 << bsr_word(vertexNum);
    edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
    edges.H = bsr_word(edges.N / edges.logN);

    edges.items = (edge_t *)malloc(edges.N * sizeof(*(edges.items)));
    for (int i = 0; i < edges.N; i++) {
        edges.items[i].value = 0;
        edges.items[i].dest = 0;
    }

    for (int i = 0; i < vertexNum; i++) {
        add_node();
    }

    for (unsigned int i = 0; i < edgeNum; i++)
    {
      unsigned int v0, v1;
      fscanf(fp, "%u%u", &v0, &v1);
      //printf("adding edge %u %u", v0, v1);
      if (v0 < v1)
        add_edge(v0, v1, 1);
      else
        add_edge(v1, v0, 1);
    }

    fclose(fp);

    // update the values of some edges

    //for (int i = 0; i < 5; i++) {
    //add_edge_update(i, i, 2);
    //}

    // print out the graph
    //printGraph();
}

template <class T>
PCSR<T>::~PCSR()
{  free(edges.items); }

template <class T>
void PCSR<T>::clear() {
  int n = 0;
  free(edges.items);
  edges.N = 2 << bsr_word(n);
  edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
  edges.H = bsr_word(edges.N / edges.logN);
}

template <class T>
vector<float> PCSR<T>::pagerank(std::vector<float> const &node_values) {
  uint64_t n = get_n();

  vector<float> output(n, 0);
  float *output_p = output.data();
  for (int i = 0; i < n; i++) {
    uint32_t start = nodes[i].beginning;
    uint32_t end = nodes[i].end;

    // get neighbors
    // start at +1 for the sentinel
    float contrib = (node_values[i] / nodes[i].num_neighbors);
    for (int j = start + 1; j < end; j++) {
      if (!is_null(edges.items[j])) {
        output_p[edges.items[j].dest] += contrib;
      }
    }
  }
  return output;
}

template <class T>
uint64_t PCSR<T>::get_n() { return nodes.size(); }

template <class T>
bool PCSR<T>::is_null(edge_t e) { return e.value == 0; }

template <class T>
vector<tuple<uint32_t, uint32_t, uint32_t>> PCSR<T>::get_edges() {
  uint64_t n = get_n();
  vector<tuple<uint32_t, uint32_t, uint32_t>> output;

  for (int i = 0; i < n; i++) {
    uint32_t start = nodes[i].beginning;
    uint32_t end = nodes[i].end;
    for (int j = start + 1; j < end; j++) {
      if (!is_null(edges.items[j])) {
        output.push_back(
            make_tuple(i, edges.items[j].dest, edges.items[j].value));
      }
    }
  }
  return output;
}

template <class T>
vector<uint32_t> PCSR<T>::get_degrees() {
  uint64_t n = get_n();
  vector<uint32_t> output;

  for (int i = 0; i < n; i++) {
        output.push_back(
            nodes[i].num_neighbors);
  }
  return output;
}

template <class T>
vector<uint32_t> PCSR<T>::bfs(uint32_t start_node) {
  uint64_t n = get_n();
  vector<uint32_t> out(n, UINT32_MAX);
  queue<uint32_t> next;
  next.push(start_node);
  out[start_node] = 0;

  while (!next.empty()) {
    uint32_t active = next.front();
    next.pop();

    uint32_t start = nodes[active].beginning;
    uint32_t end = nodes[active].end;

    // get neighbors
    // start at +1 for the sentinel
    for (int j = start + 1; j < end; j++) {
      if (!is_null(edges.items[j]) && out[edges.items[j].dest] == UINT32_MAX) {
        next.push(edges.items[j].dest);
        out[edges.items[j].dest] = out[active] + 1;
      }
    }
  }
  return out;
}

template <class T>
uint64_t PCSR<T>::get_size() {
  uint64_t size = nodes.capacity() * sizeof(node_t);
  size += edges.N * sizeof(edge_t);
  return size;
}


template <class T>
void PCSR<T>::print_array() {
  for (int i = 0; i < edges.N; i++) {
    if (is_null(edges.items[i])) {
      printf("%d-x ", i);
    } else if (is_sentinel(edges.items[i])) {
      uint32_t value = edges.items[i].value;
      if (value == UINT32_MAX) {
        value = 0;
      }
      printf("\n%d-s(%u):(%d, %d) ", i, value, nodes[value].beginning,
             nodes[value].end);
    } else {
      printf("%d-(%d, %u) ", i, edges.items[i].dest, edges.items[i].value);
    }
  }
  printf("\n\n");
}

// fix pointer from node to moved sentinel
template <class T>
void PCSR<T>::fix_sentinel(int32_t node_index, int in) {
  nodes[node_index].beginning = in;
  if (node_index > 0) {
    nodes[node_index - 1].end = in;
  }
  if (node_index == nodes.size() - 1) {
    nodes[node_index].end = edges.N - 1;
  }
}

// Evenly redistribute elements in the ofm, given a range to look into
// index: starting position in ofm structure
// len: area to redistribute
template <class T>
void PCSR<T>::redistribute(int index, int len) {
  // printf("REDISTRIBUTE: \n");
  // print_array();
  // std::vector<edge_t> space(len); //
  edge_t *space = (edge_t *)malloc(len * sizeof(*(edges.items)));
  int j = 0;

  // move all items in ofm in the range into
  // a temp array
  for (int i = index; i < index + len; i++) {
    space[j] = edges.items[i];
    // counting non-null edges
    j += (!is_null(edges.items[i]));
    // setting section to null
    edges.items[i].value = 0;
    edges.items[i].dest = 0;
  }

  // evenly redistribute for a uniform density
  double index_d = index;
  double step = ((double)len) / j;
  for (int i = 0; i < j; i++) {
    int in = index_d;

    edges.items[in] = space[i];
    if (is_sentinel(space[i])) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = space[i].value;
      if (node_index == UINT32_MAX) {
        node_index = 0;
      }
      fix_sentinel(node_index, in);
    }
    index_d += step;
  }
  free(space);
}

// null overrides sentinel
// e.g. in rebalance, we check if an edge is null
// first before copying it into temp, then fix the sentinels.
template <class T>
bool PCSR<T>::is_sentinel(edge_t e) {
  return e.dest == UINT32_MAX || e.value == UINT32_MAX;
}


template <class T>
void PCSR<T>::double_list() {
  edges.N *= 2;
  edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
  edges.H = bsr_word(edges.N / edges.logN);
  edges.items =
      (edge_t *)realloc(edges.items, edges.N * sizeof(*(edges.items)));
  for (int i = edges.N / 2; i < edges.N; i++) {
    edges.items[i].value = 0; // setting second half to null
    edges.items[i].dest = 0;  // setting second half to null
  }
  redistribute(0, edges.N);
}


template <class T>
void PCSR<T>::half_list() {
  edges.N /= 2;
  edges.logN = (1 << bsr_word(bsr_word(edges.N) + 1));
  edges.H = bsr_word(edges.N / edges.logN);
  edge_t *new_array = (edge_t *)malloc(edges.N * sizeof(*(edges.items)));
  int j = 0;
  for (int i = 0; i < edges.N * 2; i++) {
    if (!is_null(edges.items[i])) {
      new_array[j++] = edges.items[i];
    }
  }
  free(edges.items);
  edges.items = new_array;
  redistribute(0, edges.N);
}


// index is the beginning of the sequence that you want to slide right.
// notice that slide right does not not null the current spot.
// this is ok because we will be putting something in the current index
// after sliding everything to the right.
template <class T>
int PCSR<T>::slide_right(int index) {
  int rval = 0;
  edge_t el = edges.items[index];
  edges.items[index].dest = 0;
  edges.items[index].value = 0;
  index++;
  while (index < edges.N && !is_null(edges.items[index])) {
    edge_t temp = edges.items[index];
    edges.items[index] = el;
    if (!is_null(el) && is_sentinel(el)) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = el.value;
      if (node_index == UINT32_MAX) {
        node_index = 0;
      }
      fix_sentinel(node_index, index);
    }
    el = temp;
    index++;
  }
  if (!is_null(el) && is_sentinel(el)) {
    // fixing pointer of node that goes to this sentinel
    uint32_t node_index = el.value;
    if (node_index == UINT32_MAX) {
      node_index = 0;
    }
    fix_sentinel(node_index, index);
  }
  // TODO There might be an issue with this going of the end sometimes
  if (index == edges.N) {
    index--;
    slide_left(index);
    rval = -1;
    printf("slide off the end on the right, should be rare\n");
  }
  edges.items[index] = el;
  return rval;
}

// only called in slide right if it was going to go off the edge
// since it can't be full this doesn't need to worry about going off the other
// end
template <class T>
void PCSR<T>::slide_left(int index) {
  edge_t el = edges.items[index];
  edges.items[index].dest = 0;
  edges.items[index].value = 0;

  index--;
  while (index >= 0 && !is_null(edges.items[index])) {
    edge_t temp = edges.items[index];
    edges.items[index] = el;
    if (!is_null(el) && is_sentinel(el)) {
      // fixing pointer of node that goes to this sentinel
      uint32_t node_index = el.value;
      if (node_index == UINT32_MAX) {
        node_index = 0;
      }

      fix_sentinel(node_index, index);
    }
    el = temp;
    index--;
  }

  if (index == -1) {
    double_list();

    slide_right(0);
    index = 0;
  }
  if (!is_null(el) && is_sentinel(el)) {
    // fixing pointer of node that goes to this sentinel
    uint32_t node_index = el.value;
    if (node_index == UINT32_MAX) {
      node_index = 0;
    }
    fix_sentinel(node_index, index);
  }

  edges.items[index] = el;
}

template <class T>
uint32_t PCSR<T>::find_value(uint32_t src, uint32_t dest) {
  edge_t e;
  e.value = 0;
  e.dest = dest;
  uint32_t loc =
      binary_search(&edges, &e, nodes[src].beginning + 1, nodes[src].end);
  if (!is_null(edges.items[loc]) && edges.items[loc].dest == dest) {
    return edges.items[loc].value;
  } else {
    return 0;
  }
}

// NOTE: potentially don't need to return the index of the element that was
// inserted? insert elem at index returns index that the element went to (which
// may not be the same one that you put it at)
template <class T>
uint32_t PCSR<T>::insert(uint32_t index, edge_t elem, uint32_t src) {
  int node_index = find_leaf(&edges, index);
  // printf("node_index = %d\n", node_index);
  int level = edges.H;
  int len = edges.logN;

  // always deposit on the left
  if (is_null(edges.items[index])) {
    edges.items[index].value = elem.value;
    edges.items[index].dest = elem.dest;
  } else {
    // if the edge already exists in the graph, update its value
    // do not make another edge
    // return index of the edge that already exists
    if (!is_sentinel(elem) && edges.items[index].dest == elem.dest) {
      edges.items[index].value = elem.value;
      return index;
    }
    if (index == edges.N - 1) {
      // when adding to the end double then add edge
      double_list();
      node_t node = nodes[src];
      uint32_t loc_to_add =
          binary_search(&edges, &elem, node.beginning + 1, node.end);
      return insert(loc_to_add, elem, src);
    } else {
      if (slide_right(index) == -1) {
        index -= 1;
        slide_left(index);
      }
    }
    edges.items[index].value = elem.value;
    edges.items[index].dest = elem.dest;
  }

  double density = get_density(&edges, node_index, len);

  // spill over into next level up, node is completely full.
  if (density == 1) {
    node_index = find_node(node_index, len * 2);
    redistribute(node_index, len * 2);
  } else {
    // makes the last slot in a section empty so you can always slide right
    redistribute(node_index, len);
  }

  // get density of the leaf you are in
  pair_double density_b = density_bound(&edges, level);
  density = get_density(&edges, node_index, len);

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density >= density_b.y) {
    len *= 2;
    if (len <= edges.N) {
      level--;
      node_index = find_node(node_index, len);
      density_b = density_bound(&edges, level);
      density = get_density(&edges, node_index, len);
    } else {
      // if you reach the root, double the list
      double_list();

      // search from the beginning because list was doubled
      return find_elem_pointer(&edges, 0, elem);
    }
  }
  redistribute(node_index, len);

  return find_elem_pointer(&edges, node_index, elem);
}

// important: make sure start, end don't include sentinels
// returns the index of the smallest element bigger than you in the range
// [start, end) if no such element is found, returns end (because insert shifts
// everything to the right)
template <class T>
uint32_t PCSR<T>::binary_search(edge_list_t *list, edge_t *elem, uint32_t start,
                       uint32_t end) {
  while (start + 1 < end) {
    uint32_t mid = (start + end) / 2;

    edge_t item = list->items[mid];
    uint32_t change = 1;
    uint32_t check = mid;

    bool flag = true;
    while (is_null(item) && flag) {
      flag = false;
      check = mid + change;
      if (check < end) {
        flag = true;
        if (check <= end) {
          item = list->items[check];
          if (!is_null(item)) {
            break;
          } else if (check == end) {
            break;
          }
        }
      }
      check = mid - change;
      if (check >= start) {
        flag = true;
        item = list->items[check];
      }
      change++;
    }

    if (is_null(item) || start == check || end == check) {
      if (!is_null(item) && start == check && elem->dest <= item.dest) {
        return check;
      }
      return mid;
    }

    // if we found it, return
    if (elem->dest == item.dest) {
      return check;
    } else if (elem->dest < item.dest) {
      end =
          check; // if the searched for item is less than current item, set end
    } else {
      start = check;
      // otherwise, searched for item is more than current and we set start
    }
  }
  if (end < start) {
    start = end;
  }
  // handling the case where there is one element left
  // if you are leq, return start (index where elt is)
  // otherwise, return end (no element greater than you in the range)
  // printf("start = %d, end = %d, n = %d\n", start,end, list->N);
  if (elem->dest <= list->items[start].dest && !is_null(list->items[start])) {
    return start;
  }
  return end;
}


// given index, return the starting index of the leaf it is in
template <class T>
int PCSR<T>::find_leaf(edge_list_t *list, int index) {
  return (index / list->logN) * list->logN;
}


// get density of a node
template <class T>
double PCSR<T>::get_density(edge_list_t *list, int index, int len) {
  int full = 0;
  for (int i = index; i < index + len; i++) {
    full += (!is_null(list->items[i]));
  }
  double full_d = (double)full;
  return full_d / len;
}

// when adjusting the list size, make sure you're still in the
// density bound
template <class T>
pair_double PCSR<T>::density_bound(edge_list_t *list, int depth) {
  pair_double pair;

  // between 1/4 and 1/2
  // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
  // between 1/8 and 1/4
  pair.x = 1.0 / 4.0 - ((.125 * depth) / list->H);
  pair.y = 3.0 / 4.0 + ((.25 * depth) / list->H);
  return pair;
}

// return index of the edge elem
// takes in edge list and place to start looking
template <class T>
uint32_t PCSR<T>::find_elem_pointer(edge_list_t *list, uint32_t index, edge_t elem) {
  edge_t item = list->items[index];
  while (!edge_equals(item, elem)) {
    item = list->items[++index];
  }
  return index;
}

// true if e1, e2 are equals
template <class T>
bool PCSR<T>::edge_equals(edge_t e1, edge_t e2) {
  return e1.dest == e2.dest && e1.value == e2.value;
}



template <class T>
std::vector<uint32_t> PCSR<T>::sparse_matrix_vector_multiplication(std::vector<uint32_t> const &v) {
  std::vector<uint32_t> result(nodes.size(), 0);

  int num_vertices = nodes.size();

  for (int i = 0; i < num_vertices; i++) {
    // +1 to avoid sentinel

    for (uint32_t j = nodes[i].beginning + 1; j < nodes[i].end; j++) {
      result[i] += edges.items[j].value * v[edges.items[j].dest];
    }
  }
  return result;
}


template <class T>
void PCSR<T>::print_graph() {
  int num_vertices = nodes.size();
  for (int i = 0; i < num_vertices; i++) {
    // +1 to avoid sentinel
    int matrix_index = 0;

    for (uint32_t j = nodes[i].beginning + 1; j < nodes[i].end; j++) {
      if (!is_null(edges.items[j])) {
        while (matrix_index < edges.items[j].dest) {
          printf("000 ");
          matrix_index++;
        }
        printf("%03d ", edges.items[j].value);
        matrix_index++;
      }
    }
    for (uint32_t j = matrix_index; j < num_vertices; j++) {
      printf("000 ");
    }
    printf("\n");
  }
}


template <class T>
void PCSR<T>::printGraph() {
  cout<<"\nDegree Array : ";
  for (auto i: get_degrees())
    cout<< " " << i;
  // cout<< " "<< std::get<0>(i) << " "<< std::get<1>(i) << " "<< std::get<2>(i) << " ";
  cout<<"\n";
  /*
  cout<<"\nsrcPtration Array : ";
  for(unsigned int i=0;i<vertexNum;i++){
      cout<<srcPtr[i]<<" ";
  }
  */
  cout<<"\nCGraph Array : ";
  for (tuple<uint32_t, uint32_t, uint32_t> i: get_edges())
    cout<< " " << std::get<1>(i);
  // cout<< " "<< std::get<0>(i) << " "<< std::get<1>(i) << " "<< std::get<2>(i) << " ";
  cout<<"\n";
}



// add a node to the graph
template <class T>
void PCSR<T>::add_node() {
  node_t node;
  int len = nodes.size();
  edge_t sentinel;
  sentinel.dest = UINT32_MAX; // placeholder
  sentinel.value = len;       // back pointer

  if (len > 0) {
    node.beginning = nodes[len - 1].end;
    node.end = node.beginning + 1;
  } else {
    node.beginning = 0;
    node.end = 1;
    sentinel.value = UINT32_MAX;
  }
  node.num_neighbors = 0;

  nodes.push_back(node);
  insert(node.beginning, sentinel, nodes.size() - 1);
}

template <class T>
void PCSR<T>::add_edge(uint32_t src, uint32_t dest, uint32_t value) {
  if (value != 0) {
    node_t node = nodes[src];
    nodes[src].num_neighbors++;

    edge_t e;
    e.dest = dest;
    e.value = value;

    uint32_t loc_to_add =
        binary_search(&edges, &e, node.beginning + 1, node.end);
    insert(loc_to_add, e, src);
  }
}

template <class T>
void PCSR<T>::add_edge_update(uint32_t src, uint32_t dest, uint32_t value) {
  if (value != 0) {
    node_t node = nodes[src];

    edge_t e;
    e.dest = dest;
    e.value = value;

    uint32_t loc_to_add =
        binary_search(&edges, &e, node.beginning + 1, node.end);
    if (edges.items[loc_to_add].dest == dest) {
      edges.items[loc_to_add].value = value;
      return;
    }
    nodes[src].num_neighbors++;
    insert(loc_to_add, e, src);
  }
}


#endif