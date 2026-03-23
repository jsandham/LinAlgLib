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

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t GROUPSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
__global__ void thomas_pcr_multiple_wavefront_kernel(int m,
                                                     int n,
                                                     const T* __restrict__ lower,
                                                     const T* __restrict__ main,
                                                     const T* __restrict__ upper,
                                                     const T* __restrict__ B,
                                                     T* __restrict__ X)
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

    for(int it = 0; it < 3 /*iter*/; it++)
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
                                  T* __restrict__ X)
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

template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
__global__ void cr_pcr_unified_kernel(int m,
                                      int n,
                                      const T* __restrict__ lower,
                                      const T* __restrict__ main,
                                      const T* __restrict__ upper,
                                      const T* __restrict__ B,
                                      T* __restrict__ X)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Shared memory allocation (4 arrays of size n)
    __shared__ T a_shared[BLOCKSIZE];
    __shared__ T b_shared[BLOCKSIZE];
    __shared__ T c_shared[BLOCKSIZE];
    __shared__ T d_shared[BLOCKSIZE];

    // 1. Load data into LDS
    if(tid < m)
    {
        a_shared[tid] = lower[tid];
        b_shared[tid] = main[tid];
        c_shared[tid] = upper[tid];
        d_shared[tid] = B[m * bid + tid];
    }
    __syncthreads();

    // --- PHASE 1: FORWARD CYCLIC REDUCTION ---
    int stride    = 1;
    int current_m = m;

    while(current_m > PCR_SIZE)
    {
        int delta = stride;
        stride *= 2;
        int active_threads = current_m / 2;

        if(tid < active_threads)
        {
            int i     = (tid + 1) * stride - 1;
            int left  = i - delta;
            int right = i + delta;

            T w1 = a_shared[i] / b_shared[left];
            T w2 = c_shared[i] / b_shared[right];

            b_shared[i] -= w1 * c_shared[left] + w2 * a_shared[right];
            d_shared[i] -= w1 * d_shared[left] + w2 * d_shared[right];
            a_shared[i] = -w1 * a_shared[left];
            c_shared[i] = -w2 * c_shared[right];
        }
        current_m /= 2;
        __syncthreads();
    }

    // --- PHASE 2: PCR ON THE REDUCED SYSTEM ---
    // Every thread that was part of the last CR step remains active here
    if(tid < current_m)
    {
        int i        = (tid + 1) * stride - 1; // Map tid to the reduced system indices
        int p_stride = stride;

        // PCR loop: continue until the p_stride covers the whole reduced range
        while(p_stride < m)
        {
            T a_i = a_shared[i];
            T b_i = b_shared[i];
            T c_i = c_shared[i];
            T d_i = d_shared[i];

            int left  = i - p_stride;
            int right = i + p_stride;

            T alpha = 0.0f, gamma = 0.0f;
            if(left >= 0)
                alpha = -a_i / b_shared[left];
            if(right < m)
                gamma = -c_i / b_shared[right];

            __syncthreads(); // Ensure all threads have read old b_shared/d_shared values

            b_shared[i] = b_i + alpha * c_shared[left] + gamma * a_shared[right];
            d_shared[i] = d_i + alpha * d_shared[left] + gamma * d_shared[right];
            a_shared[i] = alpha * a_shared[left];
            c_shared[i] = gamma * c_shared[right];

            p_stride *= 2;
            __syncthreads();
        }
        // At this point, the PCR indices are solved.
        // We store the result in d_shared to be used as 'x' in the back-solve.
        d_shared[i] = d_shared[i] / b_shared[i];
    }
    __syncthreads();

    // --- PHASE 3: BACKWARD CYCLIC REDUCTION ---
    // We expand back out to fill the rows skipped in Phase 1
    // current_n and stride are restored from the end of Phase 1
    while(stride > 1)
    {
        int delta          = stride / 2;
        int active_threads = current_m;

        // In each step of back-substitution, we fill the 'midpoints'
        if(tid < active_threads)
        {
            int i = (tid * stride) + delta - 1;

            // Only calculate if this index wasn't already solved by PCR
            // (Standard CR logic: indices solved = indices ± delta)
            T left_x  = (i - delta >= 0) ? d_shared[i - delta] : 0.0f;
            T right_x = (i + delta < m) ? d_shared[i + delta] : 0.0f;

            d_shared[i]
                = (d_shared[i] - a_shared[i] * left_x - c_shared[i] * right_x) / b_shared[i];
        }
        stride /= 2;
        current_m *= 2;
        __syncthreads();
    }

    // 4. Final Write Out
    if(tid < m)
    {
        X[m * bid + tid] = d_shared[tid];
    }
}

// Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
__global__ void crpcr_pow2_shared_kernel2(int m,
                                          int n,
                                          const T* __restrict__ lower,
                                          const T* __restrict__ main,
                                          const T* __restrict__ upper,
                                          const T* __restrict__ B,
                                          T* __restrict__ X)
{
    static_assert((BLOCKSIZE & (BLOCKSIZE - 1)) == 0, "BLOCKSIZE must be a power of two");
    static_assert((PCR_SIZE & (PCR_SIZE - 1)) == 0, "PCR_SIZE must be a power of two");
    static_assert(PCR_SIZE <= BLOCKSIZE, "PCR_SIZE must be <= BLOCKSIZE");
    static_assert(PCR_SIZE >= 2, "PCR_SIZE must be >= 2");

    const int tid = threadIdx.x;

    const int tot_iter = static_cast<int>(log2_int((2 * BLOCKSIZE) / 2));
    const int pcr_iter = static_cast<int>(log2_int(PCR_SIZE / 2));
    const int cr_iter  = tot_iter - pcr_iter;

    int stride         = 1;
    int active_threads = BLOCKSIZE;

    // Cyclic reduction shared memory
    __shared__ T sa[2 * BLOCKSIZE];
    __shared__ T sb[2 * BLOCKSIZE];
    __shared__ T sc[2 * BLOCKSIZE];
    __shared__ T srhs[2 * BLOCKSIZE];

    // Fill cyclic reduction shared memory
    sa[tid]             = (tid < m) ? lower[tid] : static_cast<T>(0);
    sa[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? lower[tid + BLOCKSIZE] : static_cast<T>(0);
    sb[tid]             = (tid < m) ? main[tid] : static_cast<T>(1);
    sb[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? main[tid + BLOCKSIZE] : static_cast<T>(1);
    sc[tid]             = (tid < m) ? upper[tid] : static_cast<T>(0);
    sc[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? upper[tid + BLOCKSIZE] : static_cast<T>(0);
    srhs[tid]           = (tid < m) ? B[tid + m * blockIdx.x] : static_cast<T>(0);
    srhs[tid + BLOCKSIZE]
        = (tid + BLOCKSIZE < m) ? B[tid + BLOCKSIZE + m * blockIdx.x] : static_cast<T>(0);

    // The first entry of the lower diagonal and the last entry of the upper
    // diagonal should be treated as zero
    if(tid == 0)
    {
        sa[tid] = static_cast<T>(0);
    }
    if(tid == (BLOCKSIZE - 1))
    {
        sc[tid + BLOCKSIZE] = static_cast<T>(0);
    }

    __syncthreads();

    // Forward reduction using cyclic reduction
    for(int j = 0; j < cr_iter; j++)
    {
        stride *= 2;

        if(tid < active_threads)
        {
            const int index = stride * tid + stride - 1;
            const int left  = index - stride / 2;
            const int right = index + stride / 2;

            const bool left_valid  = (left >= 0);
            const bool right_valid = (right < 2 * BLOCKSIZE);

            const T k1 = left_valid ? (sa[index] / sb[left]) : static_cast<T>(0);
            const T k2 = right_valid ? (sc[index] / sb[right]) : static_cast<T>(0);

            const T a_left    = left_valid ? sa[left] : static_cast<T>(0);
            const T c_left    = left_valid ? sc[left] : static_cast<T>(0);
            const T rhs_left  = left_valid ? srhs[left] : static_cast<T>(0);
            const T a_right   = right_valid ? sa[right] : static_cast<T>(0);
            const T c_right   = right_valid ? sc[right] : static_cast<T>(0);
            const T rhs_right = right_valid ? srhs[right] : static_cast<T>(0);

            sb[index]   = sb[index] - c_left * k1 - a_right * k2;
            srhs[index] = srhs[index] - rhs_left * k1 - rhs_right * k2;
            sa[index]   = -a_left * k1;
            sc[index]   = -c_right * k2;
        }

        active_threads /= 2;

        __syncthreads();
    }

    // Parallel cyclic reduction
    const int index      = stride * tid + stride - 1;
    int       pcr_stride = stride;

    for(int j = 0; j < pcr_iter; j++)
    {
        T ta;
        T tb;
        T tc;
        T trhs;

        if(tid < PCR_SIZE)
        {
            const int right = index + pcr_stride;
            const int left  = index - pcr_stride;

            const bool left_valid  = (left >= 0);
            const bool right_valid = (right < 2 * BLOCKSIZE);

            const T a_left    = left_valid ? sa[left] : static_cast<T>(0);
            const T b_left    = left_valid ? sb[left] : static_cast<T>(0);
            const T c_left    = left_valid ? sc[left] : static_cast<T>(0);
            const T rhs_left  = left_valid ? srhs[left] : static_cast<T>(0);
            const T a_right   = right_valid ? sa[right] : static_cast<T>(0);
            const T b_right   = right_valid ? sb[right] : static_cast<T>(0);
            const T c_right   = right_valid ? sc[right] : static_cast<T>(0);
            const T rhs_right = right_valid ? srhs[right] : static_cast<T>(0);

            const T k1 = (left_valid && b_left != static_cast<T>(0)) ? (sa[index] / b_left)
                                                                     : static_cast<T>(0);
            const T k2 = (right_valid && b_right != static_cast<T>(0)) ? (sc[index] / b_right)
                                                                       : static_cast<T>(0);

            tb   = sb[index] - c_left * k1 - a_right * k2;
            trhs = srhs[index] - rhs_left * k1 - rhs_right * k2;
            ta   = -a_left * k1;
            tc   = -c_right * k2;
        }

        __syncthreads();
        if(tid < PCR_SIZE)
        {
            sb[index]   = tb;
            srhs[index] = trhs;
            sa[index]   = ta;
            sc[index]   = tc;
        }
        pcr_stride *= 2;
        __syncthreads();
    }

    if(tid < PCR_SIZE / 2)
    {
        const int index = stride * tid + stride - 1;

        // Solve 2x2 systems
        const int i         = index;
        const int j         = index + pcr_stride;
        const T   det       = static_cast<T>(1) / (sb[j] * sb[i] - sc[i] * sa[j]);
        const T   rhs_i_old = srhs[i];
        const T   rhs_j_old = srhs[j];

        srhs[i] = (sb[j] * rhs_i_old - sc[i] * rhs_j_old) * det;
        srhs[j] = (rhs_j_old * sb[i] - rhs_i_old * sa[j]) * det;
    }

    __syncthreads();

    // Backward substitution using cyclic reduction
    active_threads = PCR_SIZE;
    for(int j = 0; j < cr_iter; j++)
    {
        __syncthreads();

        if(tid < active_threads)
        {
            const int index = stride * tid + stride / 2 - 1;
            const int left  = index - stride / 2;
            const int right = index + stride / 2;

            const T x_left  = (left >= 0) ? srhs[left] : static_cast<T>(0);
            const T x_right = (right < 2 * BLOCKSIZE) ? srhs[right] : static_cast<T>(0);

            srhs[index] = (srhs[index] - sa[index] * x_left - sc[index] * x_right) / sb[index];
        }

        stride /= 2;
        active_threads *= 2;
    }

    __syncthreads();

    if(tid < m)
    {
        X[tid + m * blockIdx.x] = srhs[tid];
    }
    if(tid + BLOCKSIZE < m)
    {
        X[tid + BLOCKSIZE + m * blockIdx.x] = srhs[tid + BLOCKSIZE];
    }
}

#endif // TRIDIAGONAL_SOLVER_KERNELS_H
