//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#ifndef TRIDIAGONAL_SOLVER_KERNELS_H
#define TRIDIAGONAL_SOLVER_KERNELS_H

#include <cuda/atomic>

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t M, typename T>
__global__ void thomas_algorithm_kernel(int n,
                                        const T* __restrict__ lower_diag,
                                        const T* __restrict__ main_diag,
                                        const T* __restrict__ upper_diag,
                                        const T* __restrict__ b,
                                        T* __restrict__ x)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * BLOCKSIZE + tid;

    if(gid >= n)
    {
        return;
    }

    T c_prime[M];
    T d_prime[M];

    T main_diag_local[M];
    T lower_diag_local[M];
    T upper_diag_local[M];

#pragma unroll
    for(int i = 0; i < M; i++)
    {
        main_diag_local[i]  = main_diag[i];
        lower_diag_local[i] = lower_diag[i];
        upper_diag_local[i] = upper_diag[i];
    }

    // Each thread solves one tridiagonal system
    // Forward sweep
    c_prime[0] = upper_diag_local[0] / main_diag_local[0];
#pragma unroll
    for(int i = 1; i < M - 1; i++)
    {
        double denom = main_diag_local[i] - lower_diag_local[i] * c_prime[i - 1];
        c_prime[i]   = upper_diag_local[i] / denom;
    }

    d_prime[0] = b[M * gid + 0] / main_diag_local[0];
#pragma unroll
    for(int i = 1; i < M; i++)
    {
        double num   = b[M * gid + i] - lower_diag_local[i] * d_prime[(i - 1)];
        double denom = main_diag_local[i] - lower_diag_local[i] * c_prime[i - 1];
        d_prime[i]   = num / denom;
    }

    // Back substitution
    x[M * gid + (M - 1)] = d_prime[(M - 1)];
#pragma unroll
    for(int i = M - 2; i >= 0; i--)
    {
        x[M * gid + i] = d_prime[i] - c_prime[i] * x[M * gid + (i + 1)];
    }
}

template <uint32_t BLOCKSIZE, uint32_t MAX_M, typename T>
__global__ void thomas_algorithm_kernel2(int m,
                                         int n,
                                         const T* __restrict__ lower_diag,
                                         const T* __restrict__ main_diag,
                                         const T* __restrict__ upper_diag,
                                         const T* __restrict__ b,
                                         T* __restrict__ x)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * BLOCKSIZE + tid;

    if(gid >= n)
    {
        return;
    }

    T c_prime[MAX_M];
    T d_prime[MAX_M];

    // Each thread solves one tridiagonal system
    // Forward sweep
    c_prime[0] = upper_diag[0] / main_diag[0];
    for(int i = 1; i < m - 1; i++)
    {
        double denom = main_diag[i] - lower_diag[i] * c_prime[i - 1];
        c_prime[i]   = upper_diag[i] / denom;
    }

    d_prime[0] = b[m * gid + 0] / main_diag[0];
    for(int i = 1; i < m; i++)
    {
        double num   = b[m * gid + i] - lower_diag[i] * d_prime[(i - 1)];
        double denom = main_diag[i] - lower_diag[i] * c_prime[i - 1];
        d_prime[i]   = num / denom;
    }

    // Back substitution
    x[m * gid + (m - 1)] = d_prime[(m - 1)];
    for(int i = m - 2; i >= 0; i--)
    {
        x[m * gid + i] = d_prime[i] - c_prime[i] * x[m * gid + (i + 1)];
    }
}

#endif // TRIDIAGONAL_SOLVER_KERNELS_H
