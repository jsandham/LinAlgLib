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

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_kernels.h"

#include "../../../trace.h"

namespace linalg
{
    template <typename T>
    __host__ __device__ T abs(T val)
    {
        return val < 0 ? -val : val;
    }

    template <typename T>
    __host__ __device__ T max(T val1, T val2)
    {
        return val1 < val2 ? val2 : val1;
    }
}

#define FULL_MASK 0xffffffff

#ifdef _DEBUG
#define CHECK_CUDA(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = (call);                                                            \
        if(err != cudaSuccess)                                                               \
        {                                                                                    \
            std::cerr << "CUDA API Error: " << cudaGetErrorString(err) << " at " << __FILE__ \
                      << ":" << __LINE__ << " in function " << __func__ << std::endl;        \
            /*exit(EXIT_FAILURE);*/                                                          \
        }                                                                                    \
    } while(0)
#else
#define CHECK_CUDA(call) call
#endif

#ifdef _DEBUG
#define CHECK_CUDA_LAUNCH_ERROR()                                                                \
    do                                                                                           \
    {                                                                                            \
        cudaError_t err = cudaGetLastError();                                                    \
        if(err != cudaSuccess)                                                                   \
        {                                                                                        \
            std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) << " at " << __FILE__ \
                      << ":" << __LINE__ << " in function " << __func__ << std::endl;            \
            /*exit(EXIT_FAILURE);*/                                                              \
        }                                                                                        \
        cudaDeviceSynchronize();                                                                 \
        err = cudaGetLastError();                                                                \
        if(err != cudaSuccess)                                                                   \
        {                                                                                        \
            std::cerr << "CUDA Device Synchronization Error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << " in function " << __func__    \
                      << std::endl;                                                              \
        }                                                                                        \
        /* exit(EXIT_FAILURE); */                                                                \
    } while(0)
#else
#define CHECK_CUDA_LAUNCH_ERROR()
#endif

template <uint32_t BLOCKSIZE, typename T>
static __global__ void fill_kernel(T* __restrict__ data, size_t size, T val)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        data[tid] = val;
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void copy_kernel(T* __restrict__ dest, const T* __restrict__ src, size_t size)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        dest[tid] = src[tid];
    }
}

template <uint32_t WARPSIZE, typename T>
static __device__ void warp_reduction_sum(T* __restrict__ data, int lid)
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
static __device__ void block_reduction_sum(T* __restrict__ data, int tid)
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
static __device__ void block_reduction_max(T* __restrict__ data, int tid)
{
    if(BLOCKSIZE > 512)
    {
        if(tid < 512)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[512 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 256)
    {
        if(tid < 256)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[256 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 128)
    {
        if(tid < 128)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[128 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 64)
    {
        if(tid < 64)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[64 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 32)
    {
        if(tid < 32)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[32 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 16)
    {
        if(tid < 16)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[16 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 8)
    {
        if(tid < 8)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[8 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 4)
    {
        if(tid < 4)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[4 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 2)
    {
        if(tid < 2)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[2 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 1)
    {
        if(tid < 1)
        {
            data[tid] = linalg::max(linalg::abs(data[tid]), linalg::abs(data[1 + tid]));
        }
        __syncthreads();
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void dot_product_kernel_part1(int size,
                                                const T* __restrict__ x,
                                                const T* __restrict__ y,
                                                T* __restrict__ workspace)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    __shared__ T shared[BLOCKSIZE];

    T val = static_cast<T>(0);

    int index = gid;
    while(index < size)
    {
        val += x[index] * y[index];

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
static __global__ void dot_product_kernel_part2(T* __restrict__ workspace)
{
    int tid = threadIdx.x;

    __shared__ T shared[BLOCKSIZE];

    shared[tid] = workspace[tid];

    __syncthreads();

    block_reduction_sum<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[0] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void
    find_max_kernel_part1(int size, const T* __restrict__ x, T* __restrict__ workspace)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    __shared__ T shared[BLOCKSIZE];

    T val = static_cast<T>(0);

    int index = gid;
    while(index < size)
    {
        val = linalg::max(linalg::abs(val), linalg::abs(x[index]));

        index += BLOCKSIZE * gridDim.x;
    }

    shared[tid] = val;

    __syncthreads();

    block_reduction_max<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[bid] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void find_max_kernel_part2(T* __restrict__ workspace)
{
    int tid = threadIdx.x;

    __shared__ T shared[BLOCKSIZE];

    shared[tid] = workspace[tid];

    __syncthreads();

    block_reduction_max<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[0] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
static __global__ void csrmv_vector_kernel(int     m,
                                           int     n,
                                           int     nnz,
                                           const T alpha,
                                           const int* __restrict__ csr_row_ptr,
                                           const int* __restrict__ csr_col_ind,
                                           const T* __restrict__ csr_val,
                                           const T* __restrict__ x,
                                           const T beta,
                                           T* __restrict__ y)
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

        T sum = static_cast<T>(0);
        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T   val = csr_val[j];

            sum += x[col] * val;
        }

        warp_reduction_sum<WARPSIZE>(&sum, lid);

        if(lid == 0)
        {
            if(beta == static_cast<T>(0))
            {
                y[row] = alpha * sum;
            }
            else
            {
                y[row] = alpha * sum + beta * y[row];
            }
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
static __global__ void extract_diagonal_kernel(int m,
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

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
static __global__ void compute_residual_kernel(int m,
                                               int n,
                                               int nnz,
                                               const int* __restrict__ csr_row_ptr,
                                               const int* __restrict__ csr_col_ind,
                                               const T* __restrict__ csr_val,
                                               const T* __restrict__ x,
                                               const T* __restrict__ b,
                                               T* __restrict__ res)
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

        T sum = static_cast<T>(0);
        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T   val = csr_val[j];

            sum += x[col] * val;
        }

        warp_reduction_sum<WARPSIZE>(&sum, lid);

        if(lid == 0)
        {
            res[row] = b[row] - sum;
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
static __global__ void axpy_kernel(int size, T alpha, const T* __restrict__ x, T* __restrict__ y)
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
static __global__ void
    axpby_kernel(int size, T alpha, const T* __restrict__ x, T beta, T* __restrict__ y)
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
static __global__ void axpbypgz_kernel(int size,
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

template <uint32_t BLOCKSIZE, typename T>
static __global__ void jacobi_solve_kernel(int size,
                                           const T* __restrict__ rhs,
                                           const T* __restrict__ diag,
                                           T* __restrict__ x)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < size)
    {
        x[gid] = rhs[gid] / diag[gid];
    }
}

template <typename T>
void launch_cuda_fill_kernel(T* data, size_t size, T val)
{
    ROUTINE_TRACE("launch_cuda_fill_kernel");
    fill_kernel<256><<<((size - 1) / 256 + 1), 256>>>(data, size, val);

    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_copy_kernel(T* dest, const T* src, size_t size)
{
    ROUTINE_TRACE("launch_cuda_copy_kernel");
    copy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(dest, src, size);

    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_dot_product_kernel(int size, const T* x, const T* y, T* result)
{
    ROUTINE_TRACE("launch_cuda_dot_product_kernel");
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    dot_product_kernel_part1<256><<<256, 256>>>(size, x, y, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    dot_product_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));
}

template <typename T>
void launch_cuda_norm_inf_kernel(int size, const T* x, T* result)
{
    ROUTINE_TRACE("launch_cuda_norm_inf_kernel");
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    find_max_kernel_part1<256><<<256, 256>>>(size, x, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    find_max_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));
}

template <typename T>
void launch_cuda_csrmv_kernel(int        m,
                              int        n,
                              int        nnz,
                              const T    alpha,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              const T*   csr_val,
                              const T*   x,
                              const T    beta,
                              T*         y)
{
    ROUTINE_TRACE("launch_cuda_csrmv_kernel");

    csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_extract_diagonal_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         T*         diag)
{
    ROUTINE_TRACE("launch_cuda_extract_diagonal_kernel");
    extract_diagonal_kernel<256, 4>
        <<<((m - 1) / (256 / 4) + 1), 256>>>(m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, diag);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_compute_residual_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         const T*   x,
                                         const T*   b,
                                         T*         res)
{
    ROUTINE_TRACE("launch_cuda_compute_residual_kernel");
    compute_residual_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, x, b, res);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpy_kernel(int size, T alpha, const T* x, T* y)
{
    ROUTINE_TRACE("launch_cuda_axpy_kernel");
    axpy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpby_kernel(int size, T alpha, const T* x, T beta, T* y)
{
    ROUTINE_TRACE("launch_cuda_axpby_kernel");
    axpby_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpbypgz_kernel(int size, T alpha, const T* x, T beta, const T* y, T gamma, T* z)
{
    ROUTINE_TRACE("launch_cuda_axpbypgz_kernel");
    axpbypgz_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y, gamma, z);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_jacobi_solve_kernel(int size, const T* rhs, const T* diag, T* x)
{
    ROUTINE_TRACE("launch_cuda_jacobi_solve_kernel");
    jacobi_solve_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, rhs, diag, x);
    CHECK_CUDA_LAUNCH_ERROR();
}

template void launch_cuda_fill_kernel<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void launch_cuda_fill_kernel<int32_t>(int32_t* data, size_t size, int32_t val);
template void launch_cuda_fill_kernel<int64_t>(int64_t* data, size_t size, int64_t val);
template void launch_cuda_fill_kernel<double>(double* data, size_t size, double val);

template void launch_cuda_copy_kernel<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void launch_cuda_copy_kernel<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void launch_cuda_copy_kernel<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void launch_cuda_copy_kernel<double>(double* dest, const double* src, size_t size);

template void launch_cuda_dot_product_kernel<uint32_t>(int             size,
                                                       const uint32_t* x,
                                                       const uint32_t* y,
                                                       uint32_t*       result);
template void launch_cuda_dot_product_kernel<int32_t>(int            size,
                                                      const int32_t* x,
                                                      const int32_t* y,
                                                      int32_t*       result);
template void launch_cuda_dot_product_kernel<int64_t>(int            size,
                                                      const int64_t* x,
                                                      const int64_t* y,
                                                      int64_t*       result);
template void launch_cuda_dot_product_kernel<double>(int           size,
                                                     const double* x,
                                                     const double* y,
                                                     double*       result);

//template void launch_cuda_norm_inf_kernel<uint32_t>(int size, const uint32_t* x, uint32_t* result);
template void launch_cuda_norm_inf_kernel<int32_t>(int size, const int32_t* x, int32_t* result);
template void launch_cuda_norm_inf_kernel<int64_t>(int size, const int64_t* x, int64_t* result);
template void launch_cuda_norm_inf_kernel<double>(int size, const double* x, double* result);

template void launch_cuda_csrmv_kernel<double>(int           m,
                                               int           n,
                                               int           nnz,
                                               const double  alpha,
                                               const int*    csr_row_ptr,
                                               const int*    csr_col_ind,
                                               const double* csr_val,
                                               const double* x,
                                               const double  beta,
                                               double*       y);

template void launch_cuda_extract_diagonal_kernel<double>(int           m,
                                                          int           n,
                                                          int           nnz,
                                                          const int*    csr_row_ptr,
                                                          const int*    csr_col_ind,
                                                          const double* csr_val,
                                                          double*       diag);

template void launch_cuda_compute_residual_kernel<double>(int           m,
                                                          int           n,
                                                          int           nnz,
                                                          const int*    csr_row_ptr,
                                                          const int*    csr_col_ind,
                                                          const double* csr_val,
                                                          const double* x,
                                                          const double* b,
                                                          double*       res);

template void launch_cuda_axpy_kernel<double>(int size, double alpha, const double* x, double* y);
template void launch_cuda_axpby_kernel<double>(
    int size, double alpha, const double* x, double beta, double* y);
template void launch_cuda_axpbypgz_kernel<double>(
    int size, double alpha, const double* x, double beta, const double* y, double gamma, double* z);

template void launch_cuda_jacobi_solve_kernel<double>(int           size,
                                                      const double* rhs,
                                                      const double* diag,
                                                      double*       x);
