

/*
 *  * Functions to:
 *   1) compute the parameters(maximum, mean, and standard
 *      deviation) of the degree, out degree and in degree
 *      vectors
 *   2) compute the degree vector for undirected graphs.
 *
 */

#ifndef EDMONDS_CUH
#define EDMONDS_CUH


class Edmonds {
    public:
    Edmonds(int *IC_h,int *CP_h, int *m_h,int N);
    int max_cardinality_matching();

    private:
    int V, qh, qt;

    /*Allocate device memory for the vector CP_d */
    int *CP, *IC;
    
    int *match;
    int *q;
    int *father;
    int *base;

    bool *inq;
    bool *inb;
    bool *inp;
    int LCA(int root,int u,int v);
    void mark_blossom(int lca,int u);
    void blossom_contraction(int s,int u,int v);
    int find_augmenting_path(int s);
    int augment_path(int s,int t);
};

#endif