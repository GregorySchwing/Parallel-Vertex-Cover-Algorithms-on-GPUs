//=======================================================================
// Copyright (c) 2005 Aaron Windsor
//
// Distributed under the Boost Software License, Version 1.0. 
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//=======================================================================
#include <string>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <cassert>

#include <boost/graph/max_cardinality_matching.hpp>
#include "config.h"
#include <chrono>
using namespace boost;
#include <boost/graph/graph_traits.hpp>  
#include <boost/graph/filtered_graph.hpp>

typedef adjacency_list<vecS, vecS, undirectedS> my_graph; 


extern "C"{
             #include "bfstdcsc.h"
}

unsigned long mcm_boost(int *IC_h,int *CP_h, int N){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...

    my_graph g(N);

    // our vertices are stored in a vector, so we can refer to vertices
    // by integers in the range 0..15
	unsigned int v0, v1, k, startC, endC;

	for (unsigned int i = 0; i < N; i++)
	{
        startC = CP_h[i];
        endC = CP_h[i+1];
        for (k=startC; k<endC; k++){
            v0 = i;
            v1 = IC_h[k];
            if (v0 != v1)
                add_edge(v0,v1,g);
        }        
	}

    std::vector<graph_traits<my_graph>::vertex_descriptor> mate(N);
    for (unsigned int i = 0; i < N; i++){
        put(&mate[0],i,graph_traits<my_graph>::null_vertex());
    }
    // find the maximum cardinality matching. we'll use a checked version
    // of the algorithm, which takes a little longer than the unchecked
    // version, but has the advantage that it will return "false" if the
    // matching returned is not actually a maximum cardinality matching
    // in the graph.

    std::chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max;

    begin = std::chrono::system_clock::now(); 
    bool success = checked_edmonds_maximum_cardinality_matching(g, &mate[0]);
    assert(success);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
    printf("\nElapsed Time for Boost MCM: %f\n",elapsed_seconds_max.count());
    printf("\nBoost MCM size: %lu\n",matching_size(g, &mate[0]));
    return matching_size(g, &mate[0]);
}


unsigned long mcm_boost_headstart(int *IC_h,int *CP_h, int *m_h, int N){       // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...

    my_graph g(N);

    // our vertices are stored in a vector, so we can refer to vertices
    // by integers in the range 0..15
	unsigned int v0, v1, k, startC, endC;

	for (unsigned int i = 0; i < N; i++)
	{
        startC = CP_h[i];
        endC = CP_h[i+1];
        for (k=startC; k<endC; k++){
            v0 = i;
            v1 = IC_h[k];
            if (v0 != v1)
                add_edge(v0,v1,g);
        }        
	}
    typedef typename graph_traits<my_graph>::vertex_descriptor
        vertex_descriptor_t;
    std::vector<graph_traits<my_graph>::vertex_descriptor> mate(N);
    typedef typename graph_traits< my_graph >::vertex_iterator vertex_iterator_t;
    vertex_iterator_t vi, vi_end;
    for(boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi){
        if(m_h[*vi]<0){
            put(&mate[0], *vi, graph_traits<my_graph>::null_vertex());
        }else{
            //mate[i] = graph_traits<my_graph>::null_vertex();
            //put(&mate[0],*vi,m_h[*vi]);
            vertex_descriptor_t u = *vi;
            vertex_descriptor_t v = m_h[*vi];
            put(&mate[0],u,v);
        }
    }
    
    // find the maximum cardinality matching. we'll use a checked version
    // of the algorithm, which takes a little longer than the unchecked
    // version, but has the advantage that it will return "false" if the
    // matching returned is not actually a maximum cardinality matching
    // in the graph.

    std::chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max;

    begin = std::chrono::system_clock::now(); 
    bool success = checked_edmonds_maximum_cardinality_matching_with_custom_initial(g, &mate[0]);
    assert(success);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
    printf("\nElapsed Time for Boost MCM with Headstart: %f\n",elapsed_seconds_max.count());
    printf("\nBoost MCM size: %lu\n",matching_size(g, &mate[0]));


    begin = std::chrono::system_clock::now(); 
    bool success2 = checked_edmonds_maximum_cardinality_matching_with_greedy_initial(g, &mate[0]);
    assert(success2);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
    printf("\nElapsed Time for Boost MCM greedy: %f\n",elapsed_seconds_max.count());
    printf("\nBoost MCM size: %lu\n",matching_size(g, &mate[0]));


    begin = std::chrono::system_clock::now(); 
    bool success3 = checked_edmonds_maximum_cardinality_matching_with_empty_initial(g, &mate[0]);
    assert(success3);
    end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
    printf("\nElapsed Time for Boost MCM empty: %f\n",elapsed_seconds_max.count());
    printf("\nBoost MCM size: %lu\n",matching_size(g, &mate[0]));
    return matching_size(g, &mate[0]);
}