#include <iostream>
#include <exception>
#include <string>
#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_functions.h>

#include "mis_kernels.h"

__constant__ const uint simpleHash = 0x45d9f3b;

#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define SIMPLE_HASH(a) (((a) >> 16)^(a) * simpleHash)
#define SIMPLE_SHIFT(a) (((a) >> 16)^(a))


__device__ unsigned int h(unsigned int x){
    uint x2 = SIMPLE_HASH(x);
    x2 = SIMPLE_HASH(x2);
    x2 = SIMPLE_SHIFT(x2);
    return x2;
}

__global__ void set_L(unsigned int * CP_d, unsigned int * IC_d, int * L_d, int * c, int n){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n && !L_d[i]){
    int k;
    int start = CP_d[i];
    int end = CP_d[i+1];
    int myDegree = CP_d[i+1]-CP_d[i];
    int ni;
    int deg = CP_d[i+1]-CP_d[i];
    unsigned int hashed_i = h(i);
    for (k = start;k < end; k++){
        ni = IC_d[k];
        if (L_d[ni])
            return;
        if (deg < myDegree){
            return;
        } else if (deg == myDegree && h(ni) < hashed_i){
            return;
        }
    }
    // Add vertex to L
    L_d[i]=1;
    // Set c indicating a vertex was added.
    *c=1;
  }
}

__global__ void set_L_unmatched(unsigned int * CP_d, unsigned int * IC_d, int * L_d, int * m_d, int * c, int n){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n && !L_d[i] && m_d[i] < 3){
    int k;
    int start = CP_d[i];
    int end = CP_d[i+1];
    int myDegree = CP_d[i+1]-CP_d[i];
    int ni;
    int deg = CP_d[i+1]-CP_d[i];
    unsigned int hashed_i = h(i);
    for (k = start;k < end; k++){
        ni = IC_d[k];
        if (m_d[ni]>=3)
            continue;
        if (L_d[ni])
            return;
        if (deg < myDegree){
            return;
        } else if (deg == myDegree && h(ni) < hashed_i){
            return;
        }
    }
    // Add vertex to L
    L_d[i]=1;
    // Set c indicating a vertex was added.
    *c=1;
  }
}

