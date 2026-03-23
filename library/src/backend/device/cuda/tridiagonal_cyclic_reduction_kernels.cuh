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

#ifndef TRIDIAGONAL_CYCLIC_REDUCTION_KERNELS_H
#define TRIDIAGONAL_CYCLIC_REDUCTION_KERNELS_H

#include "common.cuh"

// Parallel cyclic reduction algorithm
template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
__global__ void thomas_pcr_wavefront_kernel(int m,
                                            int n,
                                            const T* __restrict__ lower,
                                            const T* __restrict__ main,
                                            const T* __restrict__ upper,
                                            const T* __restrict__ B,
                                            T* __restrict__ X)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(WF_SIZE / 2));
    int stride = 1;

    const int global_col = (BLOCKSIZE / WF_SIZE) * bid + wid;

    T a = (lid < m) ? lower[lid] : static_cast<T>(0);
    T b = (lid < m) ? main[lid] : static_cast<T>(1);
    T c = (lid < m) ? upper[lid] : static_cast<T>(0);
    T d = (lid < m && global_col < n) ? B[m * global_col + lid] : static_cast<T>(0);

    for(int it = 0; it < iter; it++)
    {
        const int right = lid + stride;
        const int left  = lid - stride;

        T a_left = __shfl_up_sync(0xFFFFFFFF, a, stride, WF_SIZE);
        T b_left = __shfl_up_sync(0xFFFFFFFF, b, stride, WF_SIZE);
        T c_left = __shfl_up_sync(0xFFFFFFFF, c, stride, WF_SIZE);
        T d_left = __shfl_up_sync(0xFFFFFFFF, d, stride, WF_SIZE);

        if(left < 0)
        {
            a_left = static_cast<T>(0);
            b_left = static_cast<T>(0);
            c_left = static_cast<T>(0);
            d_left = static_cast<T>(0);
        }

        T a_right = __shfl_down_sync(0xFFFFFFFF, a, stride, WF_SIZE);
        T b_right = __shfl_down_sync(0xFFFFFFFF, b, stride, WF_SIZE);
        T c_right = __shfl_down_sync(0xFFFFFFFF, c, stride, WF_SIZE);
        T d_right = __shfl_down_sync(0xFFFFFFFF, d, stride, WF_SIZE);

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

        stride *= 2;
    }

    // Solve 2x2 systems (j = lid + stride)
    // bi ci
    // aj bj
    //
    // det = bi * bj - aj * ci
    const T aj = __shfl_down_sync(0xFFFFFFFF, a, stride, WF_SIZE);
    const T bj = __shfl_down_sync(0xFFFFFFFF, b, stride, WF_SIZE);
    const T dj = __shfl_down_sync(0xFFFFFFFF, d, stride, WF_SIZE);

    if(lid < WF_SIZE / 2) // same as lid < stride
    {
        const T det = static_cast<T>(1) / (b * bj - aj * c);

        if(lid < m)
        {
            X[m * global_col + lid] = (bj * d - c * dj) * det;
        }
        if((lid + stride) < m)
        {
            X[m * global_col + lid + stride] = (dj * b - d * aj) * det;
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
                                             T* __restrict__ X)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    int iter   = static_cast<int>(log2_int(M / 2));
    int iter2  = static_cast<int>(log2_int(WF_SIZE));
    int stride = 1;

    const int global_col = (BLOCKSIZE / WF_SIZE) * bid + wid;

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
        a[i] = ((WF_SIZE * i + lid) < m) ? lower[WF_SIZE * i + lid] : static_cast<T>(0);
        b[i] = ((WF_SIZE * i + lid) < m) ? main[WF_SIZE * i + lid] : static_cast<T>(1);
        c[i] = ((WF_SIZE * i + lid) < m) ? upper[WF_SIZE * i + lid] : static_cast<T>(0);
        d[i] = ((WF_SIZE * i + lid) < m && global_col < n) ? B[m * global_col + WF_SIZE * i + lid]
                                                           : static_cast<T>(0);
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
    if(global_col <= n)
    {
        X[m * global_col + WF_SIZE * ((M / WF_SIZE) - 1) + lid] = d[(M / WF_SIZE) - 1];
        for(int i = (M / WF_SIZE) - 2; i >= 0; i--)
        {
            X[m * global_col + WF_SIZE * i + lid] = d[i];
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, uint32_t NUM_RHS, typename T>
__global__ void pcr_shared_kernel2(int m,
                                   int n,
                                   const T* __restrict__ lower,
                                   const T* __restrict__ main,
                                   const T* __restrict__ upper,
                                   const T* __restrict__ B,
                                   T* __restrict__ X)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;

    const int iter_m      = static_cast<int>(log2_int(M));
    const int iter_wfsize = static_cast<int>(log2_int(WF_SIZE));
    const int iter        = iter_m - iter_wfsize;

    int stride = 1;

    T a = (tid < m) ? lower[tid] : static_cast<T>(0);
    T b = (tid < m) ? main[tid] : static_cast<T>(1);
    T c = (tid < m) ? upper[tid] : static_cast<T>(0);

    T d[NUM_RHS];
    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        d[rhs] = (tid < m) ? B[m * (NUM_RHS * bid + rhs) + tid] : static_cast<T>(0);
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

    for(int j = 0; j < iter; j++)
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

        T d_new[NUM_RHS];
        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            const T d_left = (left >= 0) ? d_shared[left + rhs * BLOCKSIZE] : static_cast<T>(0);
            const T d_right
                = (right < BLOCKSIZE) ? d_shared[right + rhs * BLOCKSIZE] : static_cast<T>(0);

            d_new[rhs] = d[rhs] - d_left * k1 - d_right * k2;
        }

        __syncthreads();
        a_shared[tid] = a_new;
        b_shared[tid] = b_new;
        c_shared[tid] = c_new;

        a = a_new;
        b = b_new;
        c = c_new;

        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            d_shared[tid + rhs * BLOCKSIZE] = d_new[rhs];

            d[rhs] = d_new[rhs];
        }
        __syncthreads();

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
    for(int it = 0; it < iter_wfsize - 1; it++)
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

            if(((BLOCKSIZE / WF_SIZE) * lid + wid) < m)
            {
                X[m * (NUM_RHS * bid + rhs) + (BLOCKSIZE / WF_SIZE) * lid + wid]
                    = (bj * d[rhs] - c * dj) * det;
            }
            if(((BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid) < m)
            {
                X[m * (NUM_RHS * bid + rhs) + (BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid]
                    = (dj * b - d[rhs] * aj) * det;
            }
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

// Combined CR+PCR for multiple RHS: coefficient elimination is shared across all RHS
// B/X layout per block: [rhs0 (m entries), rhs1 (m entries), ...]
template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, uint32_t NUM_RHS, typename T>
__global__ void crpcr_pow2_shared_multi_rhs_kernel(int m,
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
    static_assert(NUM_RHS >= 1, "NUM_RHS must be >= 1");

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    constexpr int M_PAD = 2 * BLOCKSIZE;

    const int tot_iter = static_cast<int>(log2_int((2 * BLOCKSIZE) / 2));
    const int pcr_iter = static_cast<int>(log2_int(PCR_SIZE / 2));
    const int cr_iter  = tot_iter - pcr_iter;

    int stride         = 1;
    int active_threads = BLOCKSIZE;

    __shared__ T sa[M_PAD];
    __shared__ T sb[M_PAD];
    __shared__ T sc[M_PAD];
    __shared__ T srhs[NUM_RHS * M_PAD];

    sa[tid]             = (tid < m) ? lower[tid] : static_cast<T>(0);
    sa[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? lower[tid + BLOCKSIZE] : static_cast<T>(0);
    sb[tid]             = (tid < m) ? main[tid] : static_cast<T>(1);
    sb[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? main[tid + BLOCKSIZE] : static_cast<T>(1);
    sc[tid]             = (tid < m) ? upper[tid] : static_cast<T>(0);
    sc[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m) ? upper[tid + BLOCKSIZE] : static_cast<T>(0);

    const int rhs_block_base = bid * m * NUM_RHS;
    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        const int rhs_base = rhs * m;
        srhs[rhs * M_PAD + tid]
            = (tid < m) ? B[rhs_block_base + rhs_base + tid] : static_cast<T>(0);
        srhs[rhs * M_PAD + tid + BLOCKSIZE] = (tid + BLOCKSIZE < m)
                                                  ? B[rhs_block_base + rhs_base + tid + BLOCKSIZE]
                                                  : static_cast<T>(0);
    }

    if(tid == 0)
    {
        sa[0] = static_cast<T>(0);
    }
    if(tid == (BLOCKSIZE - 1))
    {
        sc[BLOCKSIZE + tid] = static_cast<T>(0);
    }

    __syncthreads();

    for(int j = 0; j < cr_iter; j++)
    {
        stride *= 2;

        if(tid < active_threads)
        {
            const int index = stride * tid + stride - 1;
            const int left  = index - stride / 2;
            const int right = index + stride / 2;

            const bool left_valid  = (left >= 0);
            const bool right_valid = (right < M_PAD);

            const T k1 = left_valid ? (sa[index] / sb[left]) : static_cast<T>(0);
            const T k2 = right_valid ? (sc[index] / sb[right]) : static_cast<T>(0);

            const T a_left  = left_valid ? sa[left] : static_cast<T>(0);
            const T c_left  = left_valid ? sc[left] : static_cast<T>(0);
            const T a_right = right_valid ? sa[right] : static_cast<T>(0);
            const T c_right = right_valid ? sc[right] : static_cast<T>(0);

            sb[index] = sb[index] - c_left * k1 - a_right * k2;
            sa[index] = -a_left * k1;
            sc[index] = -c_right * k2;

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                T* const rhs_vec   = &srhs[rhs * M_PAD];
                const T  rhs_left  = left_valid ? rhs_vec[left] : static_cast<T>(0);
                const T  rhs_right = right_valid ? rhs_vec[right] : static_cast<T>(0);
                rhs_vec[index]     = rhs_vec[index] - rhs_left * k1 - rhs_right * k2;
            }
        }

        active_threads /= 2;
        __syncthreads();
    }

    const int index      = stride * tid + stride - 1;
    int       pcr_stride = stride;

    for(int j = 0; j < pcr_iter; j++)
    {
        T ta;
        T tb;
        T tc;
        T trhs[NUM_RHS];

        if(tid < PCR_SIZE)
        {
            const int right = index + pcr_stride;
            const int left  = index - pcr_stride;

            const bool left_valid  = (left >= 0);
            const bool right_valid = (right < M_PAD);

            const T a_left  = left_valid ? sa[left] : static_cast<T>(0);
            const T b_left  = left_valid ? sb[left] : static_cast<T>(0);
            const T c_left  = left_valid ? sc[left] : static_cast<T>(0);
            const T a_right = right_valid ? sa[right] : static_cast<T>(0);
            const T b_right = right_valid ? sb[right] : static_cast<T>(0);
            const T c_right = right_valid ? sc[right] : static_cast<T>(0);

            const T k1 = (left_valid && b_left != static_cast<T>(0)) ? (sa[index] / b_left)
                                                                     : static_cast<T>(0);
            const T k2 = (right_valid && b_right != static_cast<T>(0)) ? (sc[index] / b_right)
                                                                       : static_cast<T>(0);

            tb = sb[index] - c_left * k1 - a_right * k2;
            ta = -a_left * k1;
            tc = -c_right * k2;

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                T* const rhs_vec   = &srhs[rhs * M_PAD];
                const T  rhs_left  = left_valid ? rhs_vec[left] : static_cast<T>(0);
                const T  rhs_right = right_valid ? rhs_vec[right] : static_cast<T>(0);
                trhs[rhs]          = rhs_vec[index] - rhs_left * k1 - rhs_right * k2;
            }
        }

        __syncthreads();
        if(tid < PCR_SIZE)
        {
            sb[index] = tb;
            sa[index] = ta;
            sc[index] = tc;
            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                srhs[rhs * M_PAD + index] = trhs[rhs];
            }
        }
        pcr_stride *= 2;
        __syncthreads();
    }

    if(tid < PCR_SIZE / 2)
    {
        const int base_index = stride * tid + stride - 1;
        const int i          = base_index;
        const int j          = base_index + pcr_stride;
        const T   det        = static_cast<T>(1) / (sb[j] * sb[i] - sc[i] * sa[j]);

        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            T* const rhs_vec = &srhs[rhs * M_PAD];
            const T  rhs_i   = rhs_vec[i];
            const T  rhs_j   = rhs_vec[j];

            rhs_vec[i] = (sb[j] * rhs_i - sc[i] * rhs_j) * det;
            rhs_vec[j] = (rhs_j * sb[i] - rhs_i * sa[j]) * det;
        }
    }

    __syncthreads();

    active_threads = PCR_SIZE;
    for(int j = 0; j < cr_iter; j++)
    {
        __syncthreads();

        if(tid < active_threads)
        {
            const int back_index = stride * tid + stride / 2 - 1;
            const int left       = back_index - stride / 2;
            const int right      = back_index + stride / 2;

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                T* const rhs_vec = &srhs[rhs * M_PAD];
                const T  x_left  = (left >= 0) ? rhs_vec[left] : static_cast<T>(0);
                const T  x_right = (right < M_PAD) ? rhs_vec[right] : static_cast<T>(0);

                rhs_vec[back_index]
                    = (rhs_vec[back_index] - sa[back_index] * x_left - sc[back_index] * x_right)
                      / sb[back_index];
            }
        }

        stride /= 2;
        active_threads *= 2;
    }

    __syncthreads();

    for(int rhs = 0; rhs < NUM_RHS; rhs++)
    {
        const int rhs_base = rhs_block_base + rhs * m;
        if(tid < m)
        {
            X[rhs_base + tid] = srhs[rhs * M_PAD + tid];
        }
        if(tid + BLOCKSIZE < m)
        {
            X[rhs_base + tid + BLOCKSIZE] = srhs[rhs * M_PAD + tid + BLOCKSIZE];
        }
    }
}

#endif // TRIDIAGONAL_CYCLIC_REDUCTION_KERNELS_H
