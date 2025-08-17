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

#ifndef AXPBY_KERNELS_H
#define AXPBY_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, typename T>
__global__ void axpy_kernel(int size, T alpha, const T* __restrict__ x, T* __restrict__ y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < size)
    {
        y[gid] = y[gid] + alpha * x[gid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void axpby_kernel(int size, T alpha, const T* __restrict__ x, T beta, T* __restrict__ y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < size)
    {
        y[gid] = alpha * x[gid] + beta * y[gid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void axpbypgz_kernel(int size,
                                T   alpha,
                                const T* __restrict__ x,
                                T beta,
                                const T* __restrict__ y,
                                T gamma,
                                T* __restrict__ z)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < size)
    {
        z[gid] = alpha * x[gid] + beta * y[gid] + gamma * z[gid];
    }
}

#endif