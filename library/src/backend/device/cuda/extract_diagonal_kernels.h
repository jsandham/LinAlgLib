//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************

#ifndef EXTRACT_DIAGONAL_KERNELS_H
#define EXTRACT_DIAGONAL_KERNELS_H

#include "common.h"

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void extract_diagonal_kernel(int m,
                                               int n,
                                               int nnz,
                                               const int* __restrict__ csr_row_ptr,
                                               const int* __restrict__ csr_col_ind,
                                               const T* __restrict__ csr_val,
                                               T* __restrict__ diag)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    //int wid = tid / WARPSIZE;

    for(int row = gid / WARPSIZE; row < m; row += (BLOCKSIZE / WARPSIZE) * gridDim.x)
    {
        int row_start = csr_row_ptr[row];
        int row_end   = csr_row_ptr[row + 1];

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T   val = csr_val[j];

            if(col == row)
            {
                diag[row] = val;
                break;
            }
        }
    }
}

#endif