/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MIS_KERNELS_GPU_H
#define MIS_KERNELS_GPU_H

//#define NR_MAX_MATCH_ROUNDS 10
__device__ unsigned int h(unsigned int x);
__global__ void set_L(unsigned int * CP_d, unsigned int * IC_d, int * L_d, int * c, int n);
__global__ void set_L_unmatched(unsigned int * CP_d, unsigned int * IC_d, int * L_d, int * m_d, int * c, int n);
#endif
