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
                                        const T* __restrict__ B,
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

    // Each thread solves one tridiagonal system
    // Forward sweep
    c_prime[0] = upper_diag[0] / main_diag[0];
    for(int i = 1; i < M - 1; i++)
    {
        const T denom = main_diag[i] - lower_diag[i] * c_prime[i - 1];
        c_prime[i]    = upper_diag[i] / denom;
    }

    d_prime[0] = B[M * gid + 0] / main_diag[0];
    for(int i = 1; i < M; i++)
    {
        const T num   = B[M * gid + i] - lower_diag[i] * d_prime[i - 1];
        const T denom = main_diag[i] - lower_diag[i] * c_prime[i - 1];
        d_prime[i]    = num / denom;
    }

    // Back substitution
    x[M * gid + (M - 1)] = d_prime[M - 1];
    for(int i = M - 2; i >= 0; i--)
    {
        x[M * gid + i] = d_prime[i] - c_prime[i] * x[M * gid + (i + 1)];
    }
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t TILE_X, uint32_t TILE_Y, typename T>
__global__ void thomas_shared_transpose_kernel1(int m,
                                                int n,
                                                const T* __restrict__ lower,
                                                const T* __restrict__ main,
                                                const T* __restrict__ upper,
                                                const T* __restrict__ B,
                                                T* __restrict__ x)
{
    constexpr uint32_t TILE_COUNT = WF_SIZE / TILE_Y;
    constexpr uint32_t PAD_TILE_X = TILE_X + 1;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    const int lid_x = lid & (TILE_X - 1);
    const int wid_x = lid / TILE_X;

    __shared__ T total_shared[(BLOCKSIZE / WF_SIZE) * PAD_TILE_X * TILE_Y * TILE_COUNT];
    T*           shared = &total_shared[PAD_TILE_X * TILE_Y * TILE_COUNT * wid];

    const T* B_ptr = &B[m * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid)];
    T*       x_ptr = &x[m * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid)];

    // Forward sweep
    // Loop over TILE_X and load 8x4=32 elements from B each iteration.
    // In total this will load 8x(4*TILE_X) elements from B
    for(int i = 0; i < TILE_COUNT; i++)
    {
        shared[PAD_TILE_X * TILE_Y * i + PAD_TILE_X * wid_x + lid_x]
            = B_ptr[m * TILE_Y * i + m * wid_x + lid_x];
    }

    T cp_local[TILE_X];
    T dp_local[TILE_X];
    T B_local[TILE_X];

    __syncthreads();
    for(int i = 0; i < TILE_X; i++)
    {
        B_local[i] = shared[PAD_TILE_X * lid + i];
    }
    __syncthreads();

    cp_local[0] = upper[0] / main[0];
    for(int i = 1; i < TILE_X; i++)
    {
        T num       = upper[i];
        T denom     = main[i] - lower[i] * cp_local[i - 1];
        cp_local[i] = num / denom;
    }

    dp_local[0] = B_local[0] / main[0];
    for(int i = 1; i < TILE_X; i++)
    {
        T num       = B_local[i] - lower[i] * dp_local[i - 1];
        T denom     = main[i] - lower[i] * cp_local[i - 1];
        dp_local[i] = num / denom;
    }

    // Backward sweep
    B_local[TILE_X - 1] = dp_local[TILE_X - 1];
    for(int i = TILE_X - 2; i >= 0; i--)
    {
        B_local[i] = dp_local[i] - cp_local[i] * B_local[i + 1];
    }

    __syncthreads();
    for(int i = 0; i < TILE_X; i++)
    {
        shared[PAD_TILE_X * lid + i] = B_local[i];
    }
    __syncthreads();

    for(int i = 0; i < TILE_X; i++)
    {
        x_ptr[m * TILE_Y * i + m * wid_x + lid_x]
            = shared[PAD_TILE_X * TILE_Y * i + PAD_TILE_X * wid_x + lid_x];
    }
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t TILE_X, uint32_t TILE_Y, typename T>
__global__ void thomas_shared_transpose_kernel2(int m,
                                                int n,
                                                const T* __restrict__ lower,
                                                const T* __restrict__ main,
                                                const T* __restrict__ upper,
                                                const T* __restrict__ B,
                                                T* __restrict__ x)
{
    constexpr uint32_t TILE_COUNT = WF_SIZE / TILE_Y;
    constexpr uint32_t PAD_TILE_X = TILE_X + 1;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    const int lid_x = lid & (TILE_X - 1);
    const int wid_x = lid / TILE_X;

    __shared__ T total_shared[(BLOCKSIZE / WF_SIZE) * PAD_TILE_X * TILE_Y * TILE_COUNT];
    T*           shared = &total_shared[PAD_TILE_X * TILE_Y * TILE_COUNT * wid];

    const T* B_ptr = &B[m * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid)];
    T*       x_ptr = &x[m * WF_SIZE * ((BLOCKSIZE / WF_SIZE) * bid + wid)];

    T cp_global[512];
    T dp_global[512];

    T temp1 = static_cast<T>(0);
    T temp2 = static_cast<T>(0);
    T temp3 = static_cast<T>(0);

    // Forward sweep
    for(int tile_start = 0; tile_start < 64; tile_start += TILE_X)
    {
        // Loop over TILE_X and load 8x4=32 elements from B each iteration.
        // In total this will load 8x(4*TILE_X) elements from B
        for(int i = 0; i < TILE_COUNT; i++)
        {
            shared[PAD_TILE_X * TILE_Y * i + PAD_TILE_X * wid_x + lid_x]
                = B_ptr[m * TILE_Y * i + m * wid_x + lid_x + tile_start];
        }

        T cp_local[TILE_X];
        T dp_local[TILE_X];
        T B_local[TILE_X];

        __syncthreads();
        for(int i = 0; i < TILE_X; i++)
        {
            B_local[i] = shared[PAD_TILE_X * lid + i];
        }
        __syncthreads();

        cp_local[0] = upper[tile_start] / (main[tile_start] - lower[tile_start] * temp1);
        for(int i = 1; i < TILE_X; i++)
        {
            T num       = upper[tile_start + i];
            T denom     = main[tile_start + i] - lower[tile_start + i] * cp_local[i - 1];
            cp_local[i] = num / denom;
        }

        dp_local[0] = (B_local[0] - lower[tile_start] * temp2)
                      / (main[tile_start] - lower[tile_start] * temp1);
        for(int i = 1; i < TILE_X; i++)
        {
            T num       = B_local[i] - lower[tile_start + i] * dp_local[i - 1];
            T denom     = main[tile_start + i] - lower[tile_start + i] * cp_local[i - 1];
            dp_local[i] = num / denom;
        }

        temp1 = cp_local[TILE_X - 1];
        temp2 = dp_local[TILE_X - 1];

        // Write cp_local and dp_local to local memory arrays
        for(int i = 0; i < TILE_X; i++)
        {
            cp_global[tile_start + i] = cp_local[i];
            dp_global[tile_start + i] = dp_local[i];
        }
    }

    // Backward sweep
    for(int tile_start = 64 - TILE_X; tile_start >= 0; tile_start -= TILE_X)
    {
        T cp_local[TILE_X];
        T dp_local[TILE_X];
        T B_local[TILE_X];

        // Read cp_local and dp_local
        for(int i = 0; i < TILE_X; i++)
        {
            cp_local[i] = cp_global[tile_start + i];
            dp_local[i] = dp_global[tile_start + i];
        }

        B_local[TILE_X - 1] = dp_local[TILE_X - 1] - cp_local[TILE_X - 1] * temp3;
        for(int i = TILE_X - 2; i >= 0; i--)
        {
            B_local[i] = dp_local[i] - cp_local[i] * B_local[i + 1];
        }

        temp3 = B_local[0];

        __syncthreads();
        for(int i = 0; i < TILE_X; i++)
        {
            shared[PAD_TILE_X * lid + i] = B_local[i];
        }
        __syncthreads();

        for(int i = 0; i < TILE_X; i++)
        {
            x_ptr[m * TILE_Y * i + m * wid_x + lid_x + tile_start]
                = shared[PAD_TILE_X * TILE_Y * i + PAD_TILE_X * wid_x + lid_x];
        }
    }
}

__device__ int log2_int(unsigned int value)
{
    return 31 - __clz(value);
}

// Parallel cyclic reduction algorithm
template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
__global__ void thomas_pcr_wavefront_kernel(int m,
                                            int n,
                                            const T* __restrict__ lower,
                                            const T* __restrict__ main,
                                            const T* __restrict__ upper,
                                            const T* __restrict__ B,
                                            T* __restrict__ X,
                                            T* __restrict__ temp_a,
                                            T* __restrict__ temp_b,
                                            T* __restrict__ temp_c,
                                            T* __restrict__ temp_d)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int stride = 1;

    T a[M / WF_SIZE];
    T b[M / WF_SIZE];
    T c[M / WF_SIZE];
    T d[M / WF_SIZE];

    for(int i = 0; i < M / WF_SIZE; i++)
    {
        a[i] = lower[WF_SIZE * i + lid];
        b[i] = main[WF_SIZE * i + lid];
        c[i] = upper[WF_SIZE * i + lid];
        d[i] = B[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
    }

    for(int it = 0; it < iter; it++)
    {
        for(int i = 0; i < M / WF_SIZE; i++)
        {
            const int right = lid + stride;
            const int left  = lid - stride;

            T a_left = __shfl_up_sync(0xFFFFFFFF, a[i], stride, WF_SIZE);
            T b_left = __shfl_up_sync(0xFFFFFFFF, b[i], stride, WF_SIZE);
            T c_left = __shfl_up_sync(0xFFFFFFFF, c[i], stride, WF_SIZE);
            T d_left = __shfl_up_sync(0xFFFFFFFF, d[i], stride, WF_SIZE);

            if(left < 0)
            {
                a_left = static_cast<T>(0);
                b_left = static_cast<T>(0);
                c_left = static_cast<T>(0);
                d_left = static_cast<T>(0);
            }

            T a_right = __shfl_down_sync(0xFFFFFFFF, a[i], stride, WF_SIZE);
            T b_right = __shfl_down_sync(0xFFFFFFFF, b[i], stride, WF_SIZE);
            T c_right = __shfl_down_sync(0xFFFFFFFF, c[i], stride, WF_SIZE);
            T d_right = __shfl_down_sync(0xFFFFFFFF, d[i], stride, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = static_cast<T>(0);
                b_right = static_cast<T>(0);
                c_right = static_cast<T>(0);
                d_right = static_cast<T>(0);
            }

            //T e = __shfl_xor_sync(0xFFFFFFFF, b[i], 1, WF_SIZE);

            const T k1 = (b_left != static_cast<T>(0)) ? a[i] / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c[i] / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b[i] - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;
            const T d_new = d[i] - d_left * k1 - d_right * k2;

            a[i] = a_new;
            b[i] = b_new;
            c[i] = c_new;
            d[i] = d_new;

            //temp_a[lid] = e;
            //temp_b[lid] = b_left;
            //temp_c[lid] = c_left;
            //temp_d[lid] = d_left;
        }

        stride *= 2;
    }

    for(int i = 0; i < M / WF_SIZE; i++)
    {
        // Solve 2x2 systems (j = lid + stride)
        // bi ci
        // aj bj
        //
        // det = bi * bj - aj * ci
        const T aj = __shfl_down_sync(0xFFFFFFFF, a[i], stride, WF_SIZE);
        const T bj = __shfl_down_sync(0xFFFFFFFF, b[i], stride, WF_SIZE);
        const T dj = __shfl_down_sync(0xFFFFFFFF, d[i], stride, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride
        {
            const T det = static_cast<T>(1) / (b[i] * bj - aj * c[i]);

            X[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid]
                = (bj * d[i] - c[i] * dj) * det;
            X[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid + stride]
                = (dj * b[i] - d[i] * aj) * det;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
__global__ void thomas_pcr_wavefront_kernel2(int m,
                                             int n,
                                             const T* __restrict__ lower,
                                             const T* __restrict__ main,
                                             const T* __restrict__ upper,
                                             const T* __restrict__ B,
                                             T* __restrict__ X,
                                             T* __restrict__ temp_a,
                                             T* __restrict__ temp_b,
                                             T* __restrict__ temp_c,
                                             T* __restrict__ temp_d)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int iter2  = static_cast<int>(log2_int(WF_SIZE));
    int stride = 1;

    T a[M / WF_SIZE];
    T b[M / WF_SIZE];
    T c[M / WF_SIZE];
    T d[M / WF_SIZE];

    T a2[M / WF_SIZE];
    T b2[M / WF_SIZE];
    T c2[M / WF_SIZE];
    T d2[M / WF_SIZE];

    for(int i = 0; i < M / WF_SIZE; i++)
    {
        a[i] = lower[WF_SIZE * i + lid];
        b[i] = main[WF_SIZE * i + lid];
        c[i] = upper[WF_SIZE * i + lid];
        d[i] = B[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
    }

    for(int it = 0; it < iter2; it++)
    {
        for(int i = 0; i < M / WF_SIZE; i++)
        {
            const int right = lid + stride;
            const int left  = lid - stride;

            const T a_left_patch = (i > 0)
                                       ? __shfl_sync(0xFFFFFFFF, a[i - 1], WF_SIZE + left, WF_SIZE)
                                       : static_cast<T>(0);
            const T b_left_patch = (i > 0)
                                       ? __shfl_sync(0xFFFFFFFF, b[i - 1], WF_SIZE + left, WF_SIZE)
                                       : static_cast<T>(0);
            const T c_left_patch = (i > 0)
                                       ? __shfl_sync(0xFFFFFFFF, c[i - 1], WF_SIZE + left, WF_SIZE)
                                       : static_cast<T>(0);
            const T d_left_patch = (i > 0)
                                       ? __shfl_sync(0xFFFFFFFF, d[i - 1], WF_SIZE + left, WF_SIZE)
                                       : static_cast<T>(0);

            const T a_right_patch
                = (i < (M / WF_SIZE - 1))
                      ? __shfl_sync(0xFFFFFFFF, a[i + 1], right - WF_SIZE, WF_SIZE)
                      : static_cast<T>(0);
            const T b_right_patch
                = (i < (M / WF_SIZE - 1))
                      ? __shfl_sync(0xFFFFFFFF, b[i + 1], right - WF_SIZE, WF_SIZE)
                      : static_cast<T>(0);
            const T c_right_patch
                = (i < (M / WF_SIZE - 1))
                      ? __shfl_sync(0xFFFFFFFF, c[i + 1], right - WF_SIZE, WF_SIZE)
                      : static_cast<T>(0);
            const T d_right_patch
                = (i < (M / WF_SIZE - 1))
                      ? __shfl_sync(0xFFFFFFFF, d[i + 1], right - WF_SIZE, WF_SIZE)
                      : static_cast<T>(0);

            T a_left = __shfl_up_sync(0xFFFFFFFF, a[i], stride, WF_SIZE);
            T b_left = __shfl_up_sync(0xFFFFFFFF, b[i], stride, WF_SIZE);
            T c_left = __shfl_up_sync(0xFFFFFFFF, c[i], stride, WF_SIZE);
            T d_left = __shfl_up_sync(0xFFFFFFFF, d[i], stride, WF_SIZE);

            if(left < 0)
            {
                a_left = a_left_patch;
                b_left = b_left_patch;
                c_left = c_left_patch;
                d_left = d_left_patch;
            }

            T a_right = __shfl_down_sync(0xFFFFFFFF, a[i], stride, WF_SIZE);
            T b_right = __shfl_down_sync(0xFFFFFFFF, b[i], stride, WF_SIZE);
            T c_right = __shfl_down_sync(0xFFFFFFFF, c[i], stride, WF_SIZE);
            T d_right = __shfl_down_sync(0xFFFFFFFF, d[i], stride, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = a_right_patch;
                b_right = b_right_patch;
                c_right = c_right_patch;
                d_right = d_right_patch;
            }

            const T k1 = (b_left != static_cast<T>(0)) ? a[i] / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c[i] / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b[i] - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;
            const T d_new = d[i] - d_left * k1 - d_right * k2;

            a[i] = a_new;
            b[i] = b_new;
            c[i] = c_new;
            d[i] = d_new;

            //temp_a[WF_SIZE * i + lid] = a_left;
            //temp_b[WF_SIZE * i + lid] = b_left;
            //temp_c[WF_SIZE * i + lid] = c_left;
            //temp_d[WF_SIZE * i + lid] = d_left;
        }

        for(int i = 0; i < M / WF_SIZE; i++)
        {
            a[i] = a2[i];
            b[i] = b2[i];
            c[i] = c2[i];
            d[i] = d2[i];
        }

        stride *= 2;
    }

    // Finish with thomas algorithm
    T c_prime[M / WF_SIZE];
    T d_prime[M / WF_SIZE];

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    for(int i = 1; i < M / WF_SIZE - 1; i++)
    {
        T num      = c[i];
        T denom    = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = num / denom;
    }

    d_prime[0] = d[0] / b[0];
    for(int i = 1; i < M / WF_SIZE; i++)
    {
        T num      = d[i] - a[i] * d_prime[i - 1];
        T denom    = b[i] - a[i] * c_prime[i - 1];
        d_prime[i] = num / denom;
    }

    // Backward sweep
    d[(M / WF_SIZE) - 1] = d_prime[(M / WF_SIZE) - 1];
    for(int i = (M / WF_SIZE) - 2; i >= 0; i--)
    {
        d[i] = d_prime[i] - c_prime[i] * d[i + 1];
    }

    // Write results to output
    X[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * ((M / WF_SIZE) - 1) + lid]
        = d[(M / WF_SIZE) - 1];
    for(int i = (M / WF_SIZE) - 2; i >= 0; i--)
    {
        X[m * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid] = d[i];
    }
}

template <uint32_t BLOCKSIZE, uint32_t GROUPSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
__global__ void thomas_pcr_multiple_wavefront_kernel(int m,
                                                     int n,
                                                     const T* __restrict__ lower,
                                                     const T* __restrict__ main,
                                                     const T* __restrict__ upper,
                                                     const T* __restrict__ B,
                                                     T* __restrict__ X,
                                                     T* __restrict__ temp_a,
                                                     T* __restrict__ temp_b,
                                                     T* __restrict__ temp_c,
                                                     T* __restrict__ temp_d)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int glid = tid & (GROUPSIZE - 1);
    const int gwid = tid / GROUPSIZE;

    const int lid = glid & (WF_SIZE - 1);
    const int wid = glid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int stride = 1;

    T a = lower[glid];
    T b = main[glid];
    T c = upper[glid];
    T d = B[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + glid];

    __shared__ T a_shared[BLOCKSIZE];
    __shared__ T b_shared[BLOCKSIZE];
    __shared__ T c_shared[BLOCKSIZE];
    __shared__ T d_shared[BLOCKSIZE];

    a_shared[GROUPSIZE * gwid + glid] = a;
    b_shared[GROUPSIZE * gwid + glid] = b;
    c_shared[GROUPSIZE * gwid + glid] = c;
    d_shared[GROUPSIZE * gwid + glid] = d;
    __syncthreads();

    for(int it = 0; it < 2 /*iter*/; it++)
    {
        const int right = lid + stride;
        const int left  = lid - stride;

        const int global_left  = glid - stride;
        const int global_right = glid + stride;

        const T a_left_patch = (left < 0 && global_left > 0)
                                   ? a_shared[GROUPSIZE * gwid + global_left]
                                   : static_cast<T>(0);
        const T b_left_patch = (left < 0 && global_left > 0)
                                   ? b_shared[GROUPSIZE * gwid + global_left]
                                   : static_cast<T>(0);
        const T c_left_patch = (left < 0 && global_left > 0)
                                   ? c_shared[GROUPSIZE * gwid + global_left]
                                   : static_cast<T>(0);
        const T d_left_patch = (left < 0 && global_left > 0)
                                   ? d_shared[GROUPSIZE * gwid + global_left]
                                   : static_cast<T>(0);

        const T a_right_patch = (right > (WF_SIZE - 1) && global_right < GROUPSIZE)
                                    ? a_shared[GROUPSIZE * gwid + global_right]
                                    : static_cast<T>(0);
        const T b_right_patch = (right > (WF_SIZE - 1) && global_right < GROUPSIZE)
                                    ? b_shared[GROUPSIZE * gwid + global_right]
                                    : static_cast<T>(0);
        const T c_right_patch = (right > (WF_SIZE - 1) && global_right < GROUPSIZE)
                                    ? c_shared[GROUPSIZE * gwid + global_right]
                                    : static_cast<T>(0);
        const T d_right_patch = (right > (WF_SIZE - 1) && global_right < GROUPSIZE)
                                    ? d_shared[GROUPSIZE * gwid + global_right]
                                    : static_cast<T>(0);
        __syncthreads();

        T a_left = __shfl_up_sync(0xFFFFFFFF, a, stride, WF_SIZE);
        T b_left = __shfl_up_sync(0xFFFFFFFF, b, stride, WF_SIZE);
        T c_left = __shfl_up_sync(0xFFFFFFFF, c, stride, WF_SIZE);
        T d_left = __shfl_up_sync(0xFFFFFFFF, d, stride, WF_SIZE);

        if(left < 0)
        {
            a_left = a_left_patch;
            b_left = b_left_patch;
            c_left = c_left_patch;
            d_left = d_left_patch;
        }

        T a_right = __shfl_down_sync(0xFFFFFFFF, a, stride, WF_SIZE);
        T b_right = __shfl_down_sync(0xFFFFFFFF, b, stride, WF_SIZE);
        T c_right = __shfl_down_sync(0xFFFFFFFF, c, stride, WF_SIZE);
        T d_right = __shfl_down_sync(0xFFFFFFFF, d, stride, WF_SIZE);

        if(right > (WF_SIZE - 1))
        {
            a_right = a_right_patch;
            b_right = b_right_patch;
            c_right = c_right_patch;
            d_right = d_right_patch;
        }

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;
        const T d_new = d - d_left * k1 - d_right * k2;

        a_shared[GROUPSIZE * gwid + glid] = a_new;
        b_shared[GROUPSIZE * gwid + glid] = b_new;
        c_shared[GROUPSIZE * gwid + glid] = c_new;
        d_shared[GROUPSIZE * gwid + glid] = d_new;

        a = a_new;
        b = b_new;
        c = c_new;
        d = d_new;
        __syncthreads();

        // temp_a[glid] = a_left;
        // temp_b[glid] = b_left;
        // temp_c[glid] = c_left;
        // temp_d[glid] = d_left;

        stride *= 2;
    }

    // // Solve 2x2 systems (j = glid + stride)
    // // bi ci
    // // aj bj
    // //
    // // det = bi * bj - aj * ci
    // const T aj
    //     = (glid < GROUPSIZE / 2) ? a_shared[GROUPSIZE * gwid + glid + stride] : static_cast<T>(0);
    // const T bj
    //     = (glid < GROUPSIZE / 2) ? b_shared[GROUPSIZE * gwid + glid + stride] : static_cast<T>(0);
    // const T dj
    //     = (glid < GROUPSIZE / 2) ? d_shared[GROUPSIZE * gwid + glid + stride] : static_cast<T>(0);
    // __syncthreads();

    // if(glid < GROUPSIZE / 2) // same as lid < stride
    // {
    //     const T det = static_cast<T>(1) / (b * bj - aj * c);

    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + glid]          = (bj * d - c * dj) * det;
    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + glid + stride] = (dj * b - d * aj) * det;
    // }

    a = a_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    b = b_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    c = c_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    d = d_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    __syncthreads();

    int stride2 = 1;
    for(int it = 0; it < 4; it++)
    {
        const int right = lid + stride2;
        const int left  = lid - stride2;

        T a_left = __shfl_up_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_left = __shfl_up_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_left = __shfl_up_sync(0xFFFFFFFF, c, stride2, WF_SIZE);
        T d_left = __shfl_up_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

        if(left < 0)
        {
            a_left = static_cast<T>(0);
            b_left = static_cast<T>(0);
            c_left = static_cast<T>(0);
            d_left = static_cast<T>(0);
        }

        T a_right = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_right = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_right = __shfl_down_sync(0xFFFFFFFF, c, stride2, WF_SIZE);
        T d_right = __shfl_down_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

        if(right > (WF_SIZE - 1))
        {
            a_right = static_cast<T>(0);
            b_right = static_cast<T>(0);
            c_right = static_cast<T>(0);
            d_right = static_cast<T>(0);
        }

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;
        const T d_new = d - d_left * k1 - d_right * k2;

        a = a_new;
        b = b_new;
        c = c_new;
        d = d_new;

        //temp_a[lid] = e;
        //temp_b[lid] = b_left;
        //temp_c[lid] = c_left;
        //temp_d[lid] = d_left;

        stride2 *= 2;
    }

    // Solve 2x2 systems (j = lid + stride2)
    // bi ci
    // aj bj
    //
    // det = bi * bj - aj * ci
    const T aj = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
    const T bj = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
    const T dj = __shfl_down_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

    if(lid < WF_SIZE / 2) // same as lid < stride2
    {
        const T det = static_cast<T>(1) / (b * bj - aj * c);

        X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + (GROUPSIZE / WF_SIZE) * lid + wid]
            = (bj * d - c * dj) * det;
        X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + (GROUPSIZE / WF_SIZE) * (lid + stride2)
          + wid]
            = (dj * b - d * aj) * det;
    }
}

template <uint32_t BLOCKSIZE, uint32_t GROUPSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
__global__ void pcr_shared_kernel(int m,
                                  int n,
                                  const T* __restrict__ lower,
                                  const T* __restrict__ main,
                                  const T* __restrict__ upper,
                                  const T* __restrict__ B,
                                  T* __restrict__ X,
                                  T* __restrict__ temp_a,
                                  T* __restrict__ temp_b,
                                  T* __restrict__ temp_c,
                                  T* __restrict__ temp_d)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int glid = tid & (GROUPSIZE - 1);
    const int gwid = tid / GROUPSIZE;

    const int lid = glid & (WF_SIZE - 1);
    const int wid = glid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int stride = 1;

    T a = lower[glid];
    T b = main[glid];
    T c = upper[glid];
    T d = B[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + glid];

    // Parallel cyclic reduction shared memory
    __shared__ T a_shared[BLOCKSIZE];
    __shared__ T b_shared[BLOCKSIZE];
    __shared__ T c_shared[BLOCKSIZE];
    __shared__ T d_shared[BLOCKSIZE];
    // __shared__ T a_shared[GROUPSIZE];
    // __shared__ T b_shared[GROUPSIZE];
    // __shared__ T c_shared[GROUPSIZE];
    // __shared__ T d_shared[BLOCKSIZE];

    // Fill parallel cyclic reduction shared memory
    a_shared[GROUPSIZE * gwid + glid] = a;
    b_shared[GROUPSIZE * gwid + glid] = b;
    c_shared[GROUPSIZE * gwid + glid] = c;
    d_shared[GROUPSIZE * gwid + glid] = d;
    // a_shared[glid]                    = a;
    // b_shared[glid]                    = b;
    // c_shared[glid]                    = c;
    // d_shared[GROUPSIZE * gwid + glid] = d;
    __syncthreads();

    for(int j = 0; j < 2 /*iter*/; j++)
    {
        const int right = glid + stride;
        const int left  = glid - stride;

        const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
        const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(0);
        const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);
        const T d_left = (left >= 0) ? d_shared[left] : static_cast<T>(0);

        const T a_right = (right < GROUPSIZE) ? a_shared[right] : static_cast<T>(0);
        const T b_right = (right < GROUPSIZE) ? b_shared[right] : static_cast<T>(0);
        const T c_right = (right < GROUPSIZE) ? c_shared[right] : static_cast<T>(0);
        const T d_right = (right < GROUPSIZE) ? d_shared[right] : static_cast<T>(0);

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;
        const T d_new = d - d_left * k1 - d_right * k2;

        __syncthreads();
        a_shared[GROUPSIZE * gwid + glid] = a_new;
        b_shared[GROUPSIZE * gwid + glid] = b_new;
        c_shared[GROUPSIZE * gwid + glid] = c_new;
        d_shared[GROUPSIZE * gwid + glid] = d_new;
        // a_shared[glid]                    = a_new;
        // b_shared[glid]                    = b_new;
        // c_shared[glid]                    = c_new;
        // d_shared[GROUPSIZE * gwid + glid] = d_new;

        a = a_new;
        b = b_new;
        c = c_new;
        d = d_new;
        __syncthreads();

        stride *= 2;
    }

    a = a_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    b = b_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    c = c_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    d = d_shared[GROUPSIZE * gwid + (GROUPSIZE / WF_SIZE) * lid + wid];
    __syncthreads();

    int stride2 = 1;
    for(int it = 0; it < 4; it++)
    {
        const int right = lid + stride2;
        const int left  = lid - stride2;

        T a_left = __shfl_up_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_left = __shfl_up_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_left = __shfl_up_sync(0xFFFFFFFF, c, stride2, WF_SIZE);
        T d_left = __shfl_up_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

        if(left < 0)
        {
            a_left = static_cast<T>(0);
            b_left = static_cast<T>(0);
            c_left = static_cast<T>(0);
            d_left = static_cast<T>(0);
        }

        T a_right = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_right = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_right = __shfl_down_sync(0xFFFFFFFF, c, stride2, WF_SIZE);
        T d_right = __shfl_down_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

        if(right > (WF_SIZE - 1))
        {
            a_right = static_cast<T>(0);
            b_right = static_cast<T>(0);
            c_right = static_cast<T>(0);
            d_right = static_cast<T>(0);
        }

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;
        const T d_new = d - d_left * k1 - d_right * k2;

        a = a_new;
        b = b_new;
        c = c_new;
        d = d_new;

        //temp_a[lid] = e;
        //temp_b[lid] = b_left;
        //temp_c[lid] = c_left;
        //temp_d[lid] = d_left;

        stride2 <<= 1; //stride2 *= 2;
    }

    // Solve 2x2 systems (j = lid + stride2)
    // bi ci
    // aj bj
    //
    // det = bi * bj - aj * ci
    const T aj = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
    const T bj = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
    const T dj = __shfl_down_sync(0xFFFFFFFF, d, stride2, WF_SIZE);

    if(lid < WF_SIZE / 2) // same as lid < stride2
    {
        const T det = static_cast<T>(1) / (b * bj - aj * c);

        X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + (GROUPSIZE / WF_SIZE) * lid + wid]
            = (bj * d - c * dj) * det;
        X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + (GROUPSIZE / WF_SIZE) * (lid + stride2)
          + wid]
            = (dj * b - d * aj) * det;
    }

    // if(glid < GROUPSIZE / 2)
    // {
    //     // Solve 2x2 systems
    //     int i   = glid;
    //     int j   = glid + stride;
    //     T   det = b_shared[j] * b_shared[i] - c_shared[i] * a_shared[j];
    //     det     = static_cast<T>(1) / det;

    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + i]
    //         = (b_shared[j] * d_shared[i] - c_shared[i] * d_shared[j]) * det;
    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + j]
    //         = (d_shared[j] * b_shared[i] - d_shared[i] * a_shared[j]) * det;
    // }
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, uint32_t NUM_RHS, typename T>
__global__ void pcr_shared_kernel2(int m,
                                   int n,
                                   const T* __restrict__ lower,
                                   const T* __restrict__ main,
                                   const T* __restrict__ upper,
                                   const T* __restrict__ B,
                                   T* __restrict__ X,
                                   T* __restrict__ temp_a,
                                   T* __restrict__ temp_b,
                                   T* __restrict__ temp_c,
                                   T* __restrict__ temp_d)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int stride = 1;

    T a = lower[tid];
    T b = main[tid];
    T c = upper[tid];

    T d[NUM_RHS];
    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        d[rhs] = B[m * NUM_RHS * bid + tid + rhs * BLOCKSIZE];
    }

    // Parallel cyclic reduction shared memory
    __shared__ T a_shared[BLOCKSIZE];
    __shared__ T b_shared[BLOCKSIZE];
    __shared__ T c_shared[BLOCKSIZE];
    __shared__ T d_shared[BLOCKSIZE * NUM_RHS];

    // Fill parallel cyclic reduction shared memory
    a_shared[tid] = a;
    b_shared[tid] = b;
    c_shared[tid] = c;
    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        d_shared[tid + rhs * BLOCKSIZE] = d[rhs];
    }
    __syncthreads();

    for(int j = 0; j < 2 /*iter*/; j++)
    {
        const int right = tid + stride;
        const int left  = tid - stride;

        const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
        const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(0);
        const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);

        const T a_right = (right < BLOCKSIZE) ? a_shared[right] : static_cast<T>(0);
        const T b_right = (right < BLOCKSIZE) ? b_shared[right] : static_cast<T>(0);
        const T c_right = (right < BLOCKSIZE) ? c_shared[right] : static_cast<T>(0);

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;

        __syncthreads();
        a_shared[tid] = a_new;
        b_shared[tid] = b_new;
        c_shared[tid] = c_new;

        a = a_new;
        b = b_new;
        c = c_new;

        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            const T d_left = (left >= 0) ? d_shared[left + rhs * BLOCKSIZE] : static_cast<T>(0);
            const T d_right
                = (right < BLOCKSIZE) ? d_shared[right + rhs * BLOCKSIZE] : static_cast<T>(0);

            const T d_new = d[rhs] - d_left * k1 - d_right * k2;

            __syncthreads();
            d_shared[tid + rhs * BLOCKSIZE] = d_new;

            d[rhs] = d_new;
            __syncthreads();
        }

        stride *= 2;
    }

    a = a_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
    b = b_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
    c = c_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        d[rhs] = d_shared[(BLOCKSIZE / WF_SIZE) * lid + wid + rhs * BLOCKSIZE];
    }
    __syncthreads();

    int stride2 = 1;
    for(int it = 0; it < 4; it++)
    {
        const int right = lid + stride2;
        const int left  = lid - stride2;

        T a_left = __shfl_up_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_left = __shfl_up_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_left = __shfl_up_sync(0xFFFFFFFF, c, stride2, WF_SIZE);

        if(left < 0)
        {
            a_left = static_cast<T>(0);
            b_left = static_cast<T>(0);
            c_left = static_cast<T>(0);
        }

        T a_right = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
        T b_right = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);
        T c_right = __shfl_down_sync(0xFFFFFFFF, c, stride2, WF_SIZE);

        if(right > (WF_SIZE - 1))
        {
            a_right = static_cast<T>(0);
            b_right = static_cast<T>(0);
            c_right = static_cast<T>(0);
        }

        const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
        const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

        const T a_new = -a_left * k1;
        const T b_new = b - c_left * k1 - a_right * k2;
        const T c_new = -c_right * k2;

        a = a_new;
        b = b_new;
        c = c_new;

        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            T d_left  = __shfl_up_sync(0xFFFFFFFF, d[rhs], stride2, WF_SIZE);
            T d_right = __shfl_down_sync(0xFFFFFFFF, d[rhs], stride2, WF_SIZE);

            if(left < 0)
            {
                d_left = static_cast<T>(0);
            }

            if(right > (WF_SIZE - 1))
            {
                d_right = static_cast<T>(0);
            }

            const T d_new = d[rhs] - d_left * k1 - d_right * k2;
            d[rhs]        = d_new;
        }

        //temp_a[lid] = e;
        //temp_b[lid] = b_left;
        //temp_c[lid] = c_left;
        //temp_d[lid] = d_left;

        stride2 <<= 1; //stride2 *= 2;
    }

    // Solve 2x2 systems (j = lid + stride2)
    // bi ci
    // aj bj
    //
    // det = bi * bj - aj * ci
    const T aj = __shfl_down_sync(0xFFFFFFFF, a, stride2, WF_SIZE);
    const T bj = __shfl_down_sync(0xFFFFFFFF, b, stride2, WF_SIZE);

    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        const T dj = __shfl_down_sync(0xFFFFFFFF, d[rhs], stride2, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride2
        {
            const T det = static_cast<T>(1) / (b * bj - aj * c);

            X[m * NUM_RHS * bid + (BLOCKSIZE / WF_SIZE) * lid + wid + rhs * BLOCKSIZE]
                = (bj * d[rhs] - c * dj) * det;
            X[m * NUM_RHS * bid + (BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid + rhs * BLOCKSIZE]
                = (dj * b - d[rhs] * aj) * det;
        }
    }

    // if(glid < GROUPSIZE / 2)
    // {
    //     // Solve 2x2 systems
    //     int i   = glid;
    //     int j   = glid + stride;
    //     T   det = b_shared[j] * b_shared[i] - c_shared[i] * a_shared[j];
    //     det     = static_cast<T>(1) / det;

    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + i]
    //         = (b_shared[j] * d_shared[i] - c_shared[i] * d_shared[j]) * det;
    //     X[m * ((BLOCKSIZE / GROUPSIZE) * bid + gwid) + j]
    //         = (d_shared[j] * b_shared[i] - d_shared[i] * a_shared[j]) * det;
    // }
}

// Cyclic reduction algorithm using shared memory
// reduce system down to a 512x512 system, solve that system with
// PCR in a single block, then back substitute to get the final solution
// That means we launch the first kernel with 512 blocks, and the second kernel with 1 block
// So if m = 65536, we launch 512 blocks with each block having blocksize = 128, and then 1
// block with blocksize = 2 * 512 = 1024 (because each block from the first stage produces 2
// unknowns for the second stage)
template <uint32_t BLOCKSIZE, typename T>
__global__ void cr_forward_sweep_kernel(int n,
                                        const T* __restrict__ lower,
                                        const T* __restrict__ main,
                                        const T* __restrict__ upper,
                                        const T* __restrict__ B,
                                        T* __restrict__ lower_pyramid,
                                        T* __restrict__ main_pyramid,
                                        T* __restrict__ upper_pyramid,
                                        T* __restrict__ rhs_pyramid,
                                        T* __restrict__ lower_spike,
                                        T* __restrict__ main_spike,
                                        T* __restrict__ upper_spike,
                                        T* __restrict__ X_spike)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * BLOCKSIZE + tid;

    // Shared memory for current working set
    __shared__ T sa[BLOCKSIZE];
    __shared__ T sb[BLOCKSIZE];
    __shared__ T sc[BLOCKSIZE];
    __shared__ T srhs[BLOCKSIZE];

    // 1. Load data from Global to Shared Memory
    if(gid < n)
    {
        sa[tid]   = lower[gid];
        sb[tid]   = main[gid];
        sc[tid]   = upper[gid];
        srhs[tid] = B[gid];
    }
    else
    {
        // Handle cases where n is not a multiple of BLOCKSIZE
        sa[tid]   = 0;
        sb[tid]   = 1;
        sc[tid]   = 0;
        srhs[tid] = 0;
    }
    __syncthreads();

    // 2. CR Forward Sweep
    int stride = 1;
    int level  = 0;

    // Reduce until only indices 0 and BLOCKSIZE-1 are active
    for(int len = BLOCKSIZE / 2; len > 1; len /= 2)
    {
        if(tid < len)
        {
            // 'elim' is the node being removed from the system at this level
            int elim  = (tid * stride * 2) + stride;
            int left  = elim - stride;
            int right = elim + stride;

            // Store the state of the node to be eliminated in the pyramid
            // We need these exact values to solve for x[elim] in the backward sweep
            int pyramid_idx            = (level * n) + (bid * BLOCKSIZE) + elim;
            lower_pyramid[pyramid_idx] = sa[elim];
            main_pyramid[pyramid_idx]  = sb[elim];
            upper_pyramid[pyramid_idx] = sc[elim];
            rhs_pyramid[pyramid_idx]   = srhs[elim];

            // Elimination Math: Use 'elim' row to modify 'left' and 'right'
            T k1 = sc[left] / sb[elim];
            T k2 = sa[right] / sb[elim];

            sb[left] -= k1 * sa[elim];
            srhs[left] -= k1 * srhs[elim];
            sc[left] = -k1 * sc[elim]; // Link 'left' to the node 'right' of elim

            sb[right] -= k2 * sc[elim];
            srhs[right] -= k2 * srhs[elim];
            sa[right] = -k2 * sa[elim]; // Link 'right' to the node 'left' of elim
        }
        stride *= 2;
        level++;
        __syncthreads();
    }

    // 3. Final Reduction: Couple Row 0 and Row (BLOCKSIZE-1)
    // After the loop, the block is reduced to just two equations.
    // We must link them so the Spike system sees a 2x2 block.
    if(tid == 0)
    {
        int first = 0;
        int last  = BLOCKSIZE - 1;

        // Take a snapshot of original values to avoid using
        // partially updated results in the second half of the math
        T f_a = sa[first], f_b = sb[first], f_c = sc[first], f_d = srhs[first];
        T l_a = sa[last], l_b = sb[last], l_c = sc[last], l_d = srhs[last];

        // Row 0 eliminates Row 'last'
        T k_f       = f_c / l_b;
        sb[first]   = f_b - k_f * l_a;
        sc[first]   = -k_f * l_c;
        srhs[first] = f_d - k_f * l_d;

        // Row 'last' eliminates Row 'first'
        T k_l      = l_a / f_b;
        sb[last]   = l_b - k_l * f_c;
        sa[last]   = -k_l * f_a;
        srhs[last] = l_d - k_l * f_d;
    }
    __syncthreads();

    // 4. Output to Spike Arrays
    if(tid == 0)
    {
        int out_idx          = 2 * bid;
        lower_spike[out_idx] = sa[0]; // Original lower[bid*BLOCKSIZE]
        main_spike[out_idx]  = sb[0];
        upper_spike[out_idx] = sc[0]; // Points to next block
        X_spike[out_idx]     = srhs[0];
    }
    else if(tid == BLOCKSIZE - 1)
    {
        int out_idx          = 2 * bid + 1;
        lower_spike[out_idx] = sa[BLOCKSIZE - 1]; // Points to prev block
        main_spike[out_idx]  = sb[BLOCKSIZE - 1];
        upper_spike[out_idx] = sc[BLOCKSIZE - 1]; // Original upper[...]
        X_spike[out_idx]     = srhs[BLOCKSIZE - 1];
    }
}

template <typename T>
__global__ void spike_solver_pcr_kernel(int num_spikes, // e.g., 512
                                        const T* __restrict__ l_spike,
                                        const T* __restrict__ m_spike,
                                        const T* __restrict__ u_spike,
                                        const T* __restrict__ d_spike,
                                        T* __restrict__ x_spike_out)
{
    const int tid = threadIdx.x;

    // Shared memory for the spike system (Size = num_spikes)
    // For 512 spikes, this is only ~8KB for doubles
    extern __shared__ char shared_mem[];
    T*                     sa = (T*)shared_mem;
    T*                     sb = sa + num_spikes;
    T*                     sc = sb + num_spikes;
    T*                     sd = sc + num_spikes;

    // 1. Load the spike system into shared memory
    if(tid < num_spikes)
    {
        sa[tid] = l_spike[tid];
        sb[tid] = m_spike[tid];
        sc[tid] = u_spike[tid];
        sd[tid] = d_spike[tid];
    }
    __syncthreads();

    // 2. PCR Algorithm
    // For 512 elements, this loop runs 9 times (2^9 = 512)
    for(int h = 1; h < num_spikes; h <<= 1)
    {
        T a = sa[tid];
        T b = sb[tid];
        T c = sc[tid];
        T d = sd[tid];

        int left  = tid - h;
        int right = tid + h;

        T k1 = 0, k2 = 0;

        if(left >= 0)
        {
            k1 = a / sb[left];
        }
        if(right < num_spikes)
        {
            k2 = c / sb[right];
        }

        __syncthreads(); // Wait for all threads to finish reading 'old' values

        // Update coefficients
        // If k1/k2 are 0 (out of bounds), the original values are preserved
        sb[tid] = b - (left >= 0 ? k1 * sc[left] : 0) - (right < num_spikes ? k2 * sa[right] : 0);
        sd[tid] = d - (left >= 0 ? k1 * sd[left] : 0) - (right < num_spikes ? k2 * sd[right] : 0);
        sa[tid] = (left >= 0 ? -k1 * sa[left] : 0);
        sc[tid] = (right < num_spikes ? -k2 * sc[right] : 0);

        __syncthreads(); // Wait for all threads to write 'new' values
    }

    // 3. Final Solution
    // After log2(N) steps, the system is diagonalized: b_i * x_i = d_i
    if(tid < num_spikes)
    {
        x_spike_out[tid] = sd[tid] / sb[tid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void cr_backward_sweep_kernel(int n,
                                         const T* __restrict__ lower_pyramid,
                                         const T* __restrict__ main_pyramid,
                                         const T* __restrict__ upper_pyramid,
                                         const T* __restrict__ rhs_pyramid,
                                         const T* __restrict__ x_spike_in,
                                         T* __restrict__ X_final)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * BLOCKSIZE + tid;

    // Shared memory to store the solving X values
    __shared__ T sx[BLOCKSIZE];

    // 1. Initialize shared memory with the boundary values from Phase 2
    if(tid == 0)
    {
        sx[0] = x_spike_in[2 * bid];
    }
    else if(tid == BLOCKSIZE - 1)
    {
        sx[BLOCKSIZE - 1] = x_spike_in[2 * bid + 1];
    }
    __syncthreads();

    // 2. Backward Sweep
    // We start from the widest stride used in the forward sweep and shrink it
    // If BLOCKSIZE is 256, the last forward len was 2, so we start backwards from there.
    int stride = BLOCKSIZE / 2;
    int level  = (int)log2f(BLOCKSIZE) - 2; // Match the level count from forward sweep

    for(int len = 2; len <= BLOCKSIZE / 2; len <<= 1)
    {
        // Only threads representing nodes to be 'filled in' are active
        if(tid < len)
        {
            int elim  = (tid * stride * 2) + stride;
            int left  = elim - stride;
            int right = elim + stride;

            // Load the coefficients that were used to eliminate this node
            int pyramid_idx = (level * n) + (bid * BLOCKSIZE) + elim;

            T a = lower_pyramid[pyramid_idx];
            T b = main_pyramid[pyramid_idx];
            T c = upper_pyramid[pyramid_idx];
            T d = rhs_pyramid[pyramid_idx];

            // Solve for x[elim]:
            // b*x_elim + a*x_left + c*x_right = d
            // x_elim = (d - a*x_left - c*x_right) / b
            sx[elim] = (d - a * sx[left] - c * sx[right]) / b;
        }
        stride >>= 1;
        level--;
        __syncthreads();
    }

    // 3. Final Step: Fill the Level 0 gaps
    // The loop above fills the tree, but we have one last set of
    // original 'odd' nodes (stride 1) to solve.
    if(tid < (BLOCKSIZE / 2))
    {
        int elim  = (tid * 2) + 1;
        int left  = elim - 1;
        int right = elim + 1;

        // Level 0 of pyramid
        int pyramid_idx = (bid * BLOCKSIZE) + elim;

        T a = lower_pyramid[pyramid_idx];
        T b = main_pyramid[pyramid_idx];
        T c = upper_pyramid[pyramid_idx];
        T d = rhs_pyramid[pyramid_idx];

        sx[elim] = (d - a * sx[left] - c * sx[right]) / b;
    }
    __syncthreads();

    // 4. Write final results to global memory
    if(gid < n)
    {
        X_final[gid] = sx[tid];
    }
}

#endif // TRIDIAGONAL_SOLVER_KERNELS_H
