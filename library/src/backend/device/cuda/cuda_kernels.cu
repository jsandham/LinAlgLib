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

#include <iostream>

#include "cuda_kernels.h"

#define FULL_MASK 0xffffffff

#define CHECK_CUDA(status)                                                   \
if(status != cudaSuccess)                                                    \
{                                                                            \
    std::cout << "Error: Cuda call returned status " << status << std::endl; \
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void fill_kernel(T* data, size_t size, T val)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        data[tid] = val;
    }
}

template <uint32_t WARPSIZE, typename T>
static __device__ void warp_reduction_sum(T* data, int lid)
{
    if(WARPSIZE > 16)
    {
        *data += __shfl_down_sync(FULL_MASK, *data, 16);       
    }
    if(WARPSIZE > 8)
    {
        *data += __shfl_down_sync(FULL_MASK, *data, 8);
    }
    if(WARPSIZE > 4)
    {
        *data += __shfl_down_sync(FULL_MASK, *data, 4);
    }
    if(WARPSIZE > 2)
    {
        *data += __shfl_down_sync(FULL_MASK, *data, 2);
    }
    if(WARPSIZE > 1)
    {
        *data += __shfl_down_sync(FULL_MASK, *data, 1);
    }
}


template <uint32_t BLOCKSIZE, typename T>
static __device__ void block_reduction_sum(T* data, int tid)
{
    if(BLOCKSIZE > 512)
    {
        if(tid < 512)
        {
            data[tid] += data[512 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 256)
    {
        if(tid < 256)
        {
            data[tid] += data[256 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 128)
    {
        if(tid < 128)
        {
            data[tid] += data[128 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 64)
    {
        if(tid < 64)
        {
            data[tid] += data[64 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 32)
    {
        if(tid < 32)
        {
            data[tid] += data[32 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 16)
    {
        if(tid < 16)
        {
            data[tid] += data[16 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 8)
    {
        if(tid < 8)
        {
            data[tid] += data[8 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 4)
    {
        if(tid < 4)
        {
            data[tid] += data[4 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 2)
    {
        if(tid < 2)
        {
            data[tid] += data[2 + tid];
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 1)
    {
        if(tid < 1)
        {
            data[tid] += data[1 + tid];
        }
        __syncthreads();
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void dot_product_kernel_part1(int size, const T* x, const T* y, T* workspace)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    __shared__ T shared[BLOCKSIZE];

    T val = static_cast<T>(0);

    int index = gid;
    while(index < size)
    {
        val = x[index] * y[index];

        index += BLOCKSIZE * gridDim.x;
    }

    shared[tid] = val;

    __syncthreads();

    block_reduction_sum<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[bid] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void dot_product_kernel_part2(const T* workspace, T* res)
{
    int tid = threadIdx.x;

    __shared__ T shared[BLOCKSIZE];

    shared[tid] = workspace[tid];

    __syncthreads();

    block_reduction_sum<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        *res = workspace[0];
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
static __global__ void csrmv_vector_kernel(int m, int n, int nnz, const T* alpha, const int* csr_row_ptr, const int* csr_col_ind, 
    const T* csr_val, const T* x, const T* beta, T* y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    //int wid = tid / WARPSIZE;

    for(int row = gid / WARPSIZE; row < m; row += (BLOCKSIZE / WARPSIZE) * gridDim.x)
    {
        int row_start = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];

        T sum = static_cast<T>(0);
        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T val = csr_val[j];

            sum += x[col] * val;
        }

        warp_reduction_sum<WARPSIZE>(&sum, lid);

        if(lid == 0)
        {
            if(*beta == static_cast<T>(0))
            {
                y[row] = *alpha * sum;
            }
            else
            {
                y[row] = *alpha * sum + *beta * y[row];
            }
        }
    }
}

template <typename T> 
void launch_cuda_fill_kernel(T* data, size_t size, T val)
{
    fill_kernel<256><<<((size - 1) / 256 + 1), 256>>>(data, size, val);
}

template void launch_cuda_fill_kernel<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void launch_cuda_fill_kernel<int32_t>(int32_t* data, size_t size, int32_t val);
template void launch_cuda_fill_kernel<int64_t>(int64_t* data, size_t size, int64_t val);
template void launch_cuda_fill_kernel<double>(double* data, size_t size, double val);

template <typename T>
void launch_cuda_dot_product_kernel(int size, const T* x, const T* y, T* result)
{
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    dot_product_kernel_part1<256><<<256, 256>>>(size, x, y, workspace);

    dot_product_kernel_part2<256><<<1, 256>>>(workspace, result);

    CHECK_CUDA(cudaFree(workspace));
}

template void launch_cuda_dot_product_kernel<uint32_t>(int size, const uint32_t* x, const uint32_t* y, uint32_t* result);
template void launch_cuda_dot_product_kernel<int32_t>(int size, const int32_t* x, const int32_t* y, int32_t* result);
template void launch_cuda_dot_product_kernel<int64_t>(int size, const int64_t* x, const int64_t* y, int64_t* result);
template void launch_cuda_dot_product_kernel<double>(int size, const double* x, const double* y, double* result);

template <typename T>
void launch_cuda_csrmv_kernel(int m, int n, int nnz, const T* alpha, const int* csr_row_ptr, 
                            const int* csr_col_ind, const T* csr_val, const T* x, const T* beta, T* y)
{
    csrmv_vector_kernel<256, 32><<<((m - 1) / 256 + 1), 256>>>(m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
}

template void launch_cuda_csrmv_kernel<double>(int m, int n, int nnz, const double* alpha, const int* csr_row_ptr, 
                            const int* csr_col_ind, const double* csr_val, const double* x, const double* beta, double* y);