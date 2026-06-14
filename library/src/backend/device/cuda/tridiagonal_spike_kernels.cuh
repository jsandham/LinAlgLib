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

#ifndef TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H
#define TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H

#include <assert.h>

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void data_marshaling_kernel(int m,
                                       int m_pad,
                                       const T* __restrict__ lower,
                                       const T* __restrict__ main,
                                       const T* __restrict__ upper,
                                       T* __restrict__ lower_pad,
                                       T* __restrict__ main_pad,
                                       T* __restrict__ upper_pad)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int gwid = gid / (m_pad / BLOCKDIM);
    const int glid = gid % (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    lower_pad[gid]
        = (BLOCKDIM * glid + gwid < m) ? lower[BLOCKDIM * glid + gwid] : static_cast<T>(0);
    main_pad[gid] = (BLOCKDIM * glid + gwid < m) ? main[BLOCKDIM * glid + gwid] : static_cast<T>(1);
    upper_pad[gid]
        = (BLOCKDIM * glid + gwid < m) ? upper[BLOCKDIM * glid + gwid] : static_cast<T>(0);
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void
    data_marshaling_B_kernel(int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int gwid = gid / (m_pad / BLOCKDIM);
    const int glid = gid % (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    for(int batch = blockIdx.y; batch < n; batch += 32768)
    {
        B_pad[gid + m_pad * batch]
        = (BLOCKDIM * glid + gwid < m) ? B[BLOCKDIM * glid + gwid + m * batch] : static_cast<T>(0);
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void
    data_marshaling_kernel2(int m, int m_pad, int n, const T* __restrict__ B_pad, T* __restrict__ B)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int gwid = gid / (m_pad / BLOCKDIM);
    const int glid = gid % (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    if(BLOCKDIM * glid + gwid < m)
    {
        for(int batch = blockIdx.y; batch < n; batch += 32768)
        {
            B[BLOCKDIM * glid + gwid + m * batch] = B_pad[gid + m_pad * batch];
        }
    }
}

template <typename T>
__host__ __device__ bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
{
    double kappa = double(0.5) * (sqrt(double(5.0)) - double(1.0));

    double sigma = double(0);
    sigma        = max(double(abs(ak_1)), double(abs(ak_2)));
    sigma        = max(double(abs(bk_1)), sigma);
    sigma        = max(double(abs(ck)), sigma);
    sigma        = max(double(abs(ck_1)), sigma);

    return abs(bk) * sigma >= kappa * abs(ak_1 * ck);
}

template <int WORDS>
struct PivotMask
{
    unsigned int bits[WORDS];

    // Sets bit k to 0 to record a 1x1 pivot at row k.
    __device__ __forceinline__ void set_pivoting_to_1x1(int k)
    {
        bits[k >> 5] &= ~(1u << (k & 31));
    }

    // Sets bit k to 1 to record 2x2 pivoting at row k.
    __device__ __forceinline__ void set_pivoting_to2x2(int k)
    {
        bits[k >> 5] |= (1u << (k & 31));
    }

    // Returns 1 if row k used 1x1 pivoting, 2 if row k is part of a 2x2 pivot.
    __device__ __forceinline__ int get_pivoting(int k) const
    {
        return ((bits[k >> 5] >> (k & 31)) & 1u) ? 2 : 1;
    }
};

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void LBMT_solve_wvmt_kernel(int m_pad,
                                       const T* __restrict__ lower,
                                       const T* __restrict__ main,
                                       const T* __restrict__ upper,
                                       T* __restrict__ w,
                                       T* __restrict__ v,
                                       T* __restrict__ mt)
{
    static_assert(BLOCKDIM >= 2);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int nblocks = m_pad / BLOCKDIM;

    if(gid >= nblocks)
    {
        return;
    }

    T bk = main[gid];

    constexpr int               PIVOT_MASK_WORDS = (BLOCKDIM + 31) / 32;
    PivotMask<PIVOT_MASK_WORDS> pivot_mask;

    w[gid]                            = lower[gid];
    v[gid + (BLOCKDIM - 1) * nblocks] = upper[gid + (BLOCKDIM - 1) * nblocks];

    int k = 0;
    while(k < BLOCKDIM)
    {
        T ck   = upper[nblocks * k + gid];
        T ck_1 = (k < (BLOCKDIM - 1)) ? upper[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T bk_1 = (k < (BLOCKDIM - 1)) ? main[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_1 = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_2 = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + gid] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (BLOCKDIM - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            T wk = w[nblocks * k + gid];
            T vk = v[nblocks * k + gid];

            w[nblocks * k + gid]  = wk * inv_bk;
            v[nblocks * k + gid]  = vk * inv_bk;
            mt[nblocks * k + gid] = ck * inv_bk;

            pivot_mask.set_pivoting_to_1x1(k);

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid] += -ak_1 * wk * inv_bk;
            }

            if(k < (BLOCKDIM - 1))
            {
                bk_1 = bk_1 - ak_1 * ck * inv_bk;
            }

            bk = bk_1;

            k += 1;
        }
        else
        {
            const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

            T wk   = w[nblocks * k + gid];
            T wk_1 = w[nblocks * (k + 1) + gid];
            T vk   = v[nblocks * k + gid];
            T vk_1 = v[nblocks * (k + 1) + gid];

            w[nblocks * k + gid]  = (bk_1 * wk - ck * wk_1) * det;
            v[nblocks * k + gid]  = (bk_1 * vk - ck * vk_1) * det;
            mt[nblocks * k + gid] = -ck * ck_1 * det;

            pivot_mask.set_pivoting_to2x2(k);

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid]  = (-ak_1 * wk + bk * wk_1) * det;
                v[nblocks * (k + 1) + gid]  = (-ak_1 * vk + bk * vk_1) * det;
                mt[nblocks * (k + 1) + gid] = bk * ck_1 * det;

                pivot_mask.set_pivoting_to2x2(k + 1);
            }

            T bk_2 = static_cast<T>(0);

            if(k < (BLOCKDIM - 2))
            {
                w[nblocks * (k + 2) + gid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
            }

            if(k < (BLOCKDIM - 2))
            {
                bk_2 = main[nblocks * (k + 2) + gid];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= pivot_mask.get_pivoting(k);

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot_mask.get_pivoting(k) == 1)
        {
            const T tmp = mt[nblocks * k + gid];

            w[nblocks * k + gid] += -tmp * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp * v[nblocks * (k + 1) + gid];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[nblocks * k + gid];
            const T tmp2 = mt[nblocks * (k - 1) + gid];

            w[nblocks * k + gid] += -tmp1 * w[nblocks * (k + 1) + gid];
            w[nblocks * (k - 1) + gid] += -tmp2 * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp1 * v[nblocks * (k + 1) + gid];
            v[nblocks * (k - 1) + gid] += -tmp2 * v[nblocks * (k + 1) + gid];

            k -= 2;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void LBMT_solve_rhs_kernel(int m_pad,
                                      int n,
                                      const T* __restrict__ lower,
                                      const T* __restrict__ main,
                                      const T* __restrict__ upper,
                                      const T* __restrict__ mt,
                                      T* __restrict__ rhs)
{
    static_assert(BLOCKDIM >= 2);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int nblocks = m_pad / BLOCKDIM;

    if(gid >= nblocks)
    {
        return;
    }

    T bk = main[gid];

    constexpr int               PIVOT_MASK_WORDS = (BLOCKDIM + 31) / 32;
    PivotMask<PIVOT_MASK_WORDS> pivot_mask;

    int k = 0;
    while(k < BLOCKDIM)
    {
        T ck   = upper[nblocks * k + gid];
        T ck_1 = (k < (BLOCKDIM - 1)) ? upper[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T bk_1 = (k < (BLOCKDIM - 1)) ? main[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_1 = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_2 = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + gid] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (BLOCKDIM - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            pivot_mask.set_pivoting_to_1x1(k);

            // L * B * x = y
            T rhsk = rhs[nblocks * k + gid + m_pad * blockIdx.y] * inv_bk;

            rhs[nblocks * k + gid + m_pad * blockIdx.y] = rhsk;

            if(k < (BLOCKDIM - 1))
            {
                rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y] += -(ak_1 * rhsk);

                bk_1 = bk_1 - ak_1 * ck * inv_bk;
            }

            bk = bk_1;

            k += 1;
        }
        else
        {
            const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

            pivot_mask.set_pivoting_to2x2(k);

            if(k < (BLOCKDIM - 1))
            {
                pivot_mask.set_pivoting_to2x2(k + 1);
            }

            T bk_2 = static_cast<T>(0);

            // |bk   ck  ||xk  |   |rhsk   |
            // |ak_1 bk_1||xk_1| = |rhsk _1|
            //
            //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
            //                                  |-ak_1  bk |

            // L * B * x = y
            T rhsk   = rhs[nblocks * k + gid + m_pad * blockIdx.y] * det;
            T rhsk_1 = rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y] * det;

            rhs[nblocks * k + gid + m_pad * blockIdx.y]       = (bk_1 * rhsk - ck * rhsk_1);
            rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y] = (-ak_1 * rhsk + bk * rhsk_1);

            if(k < (BLOCKDIM - 2))
            {
                rhs[nblocks * (k + 2) + gid + m_pad * blockIdx.y]
                    += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                bk_2 = main[nblocks * (k + 2) + gid];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= pivot_mask.get_pivoting(k);

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot_mask.get_pivoting(k) == 1)
        {
            const T tmp = mt[nblocks * k + gid];

            rhs[nblocks * k + gid + m_pad * blockIdx.y]
                += -tmp * rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[nblocks * k + gid];
            const T tmp2 = mt[nblocks * (k - 1) + gid];

            rhs[nblocks * k + gid + m_pad * blockIdx.y]
                += -tmp1 * rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y];
            rhs[nblocks * (k - 1) + gid + m_pad * blockIdx.y]
                += -tmp2 * rhs[nblocks * (k + 1) + gid + m_pad * blockIdx.y];

            k -= 2;
        }
    }
}













































template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void LBMT_solve_kernel(int m_pad,
                                  int n,
                                  const T* __restrict__ lower,
                                  const T* __restrict__ main,
                                  const T* __restrict__ upper,
                                  T* __restrict__ w,
                                  T* __restrict__ v,
                                  T* __restrict__ mt,
                                  T* __restrict__ rhs)
{
    static_assert(BLOCKDIM >= 2);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int nblocks = m_pad / BLOCKDIM;

    if(gid >= nblocks)
    {
        return;
    }

    T bk = main[gid];

    constexpr int               PIVOT_MASK_WORDS = (BLOCKDIM + 31) / 32;
    PivotMask<PIVOT_MASK_WORDS> pivot_mask;

    w[gid]                            = lower[gid];
    v[gid + (BLOCKDIM - 1) * nblocks] = upper[gid + (BLOCKDIM - 1) * nblocks];

    int k = 0;
    while(k < BLOCKDIM)
    {
        T ck   = upper[nblocks * k + gid];
        T ck_1 = (k < (BLOCKDIM - 1)) ? upper[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T bk_1 = (k < (BLOCKDIM - 1)) ? main[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_1 = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + gid] : static_cast<T>(0);
        T ak_2 = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + gid] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (BLOCKDIM - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            T wk = w[nblocks * k + gid];
            T vk = v[nblocks * k + gid];

            w[nblocks * k + gid]  = wk * inv_bk;
            v[nblocks * k + gid]  = vk * inv_bk;
            mt[nblocks * k + gid] = ck * inv_bk;

            pivot_mask.set_pivoting_to_1x1(k);

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid] += -ak_1 * wk * inv_bk;
            }

            // L * B * x = y
            T rhsk = rhs[nblocks * k + gid] * inv_bk;

            rhs[nblocks * k + gid] = rhsk;

            if(k < (BLOCKDIM - 1))
            {
                rhs[nblocks * (k + 1) + gid] += -(ak_1 * rhsk);

                bk_1 = bk_1 - ak_1 * ck * inv_bk;
            }

            bk = bk_1;

            k += 1;
        }
        else
        {
            const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

            T wk   = w[nblocks * k + gid];
            T wk_1 = w[nblocks * (k + 1) + gid];
            T vk   = v[nblocks * k + gid];
            T vk_1 = v[nblocks * (k + 1) + gid];

            w[nblocks * k + gid]  = (bk_1 * wk - ck * wk_1) * det;
            v[nblocks * k + gid]  = (bk_1 * vk - ck * vk_1) * det;
            mt[nblocks * k + gid] = -ck * ck_1 * det;

            pivot_mask.set_pivoting_to2x2(k);

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid]  = (-ak_1 * wk + bk * wk_1) * det;
                v[nblocks * (k + 1) + gid]  = (-ak_1 * vk + bk * vk_1) * det;
                mt[nblocks * (k + 1) + gid] = bk * ck_1 * det;

                pivot_mask.set_pivoting_to2x2(k + 1);
            }

            T bk_2 = static_cast<T>(0);

            if(k < (BLOCKDIM - 2))
            {
                w[nblocks * (k + 2) + gid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
            }

            // |bk   ck  ||xk  |   |rhsk   |
            // |ak_1 bk_1||xk_1| = |rhsk _1|
            //
            //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
            //                                  |-ak_1  bk |

            // L * B * x = y
            T rhsk   = rhs[nblocks * k + gid] * det;
            T rhsk_1 = rhs[nblocks * (k + 1) + gid] * det;

            rhs[nblocks * k + gid]       = (bk_1 * rhsk - ck * rhsk_1);
            rhs[nblocks * (k + 1) + gid] = (-ak_1 * rhsk + bk * rhsk_1);

            if(k < (BLOCKDIM - 2))
            {
                rhs[nblocks * (k + 2) + gid]
                    += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                bk_2 = main[nblocks * (k + 2) + gid];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= pivot_mask.get_pivoting(k);

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot_mask.get_pivoting(k) == 1)
        {
            const T tmp = mt[nblocks * k + gid];

            w[nblocks * k + gid] += -tmp * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp * v[nblocks * (k + 1) + gid];
            rhs[nblocks * k + gid]
                += -tmp * rhs[nblocks * (k + 1) + gid];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[nblocks * k + gid];
            const T tmp2 = mt[nblocks * (k - 1) + gid];

            w[nblocks * k + gid] += -tmp1 * w[nblocks * (k + 1) + gid];
            w[nblocks * (k - 1) + gid] += -tmp2 * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp1 * v[nblocks * (k + 1) + gid];
            v[nblocks * (k - 1) + gid] += -tmp2 * v[nblocks * (k + 1) + gid];
            rhs[nblocks * k + gid]
                += -tmp1 * rhs[nblocks * (k + 1) + gid];
            rhs[nblocks * (k - 1) + gid]
                += -tmp2 * rhs[nblocks * (k + 1) + gid];

            k -= 2;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void fill_s_matrix_kernel(int m_pad,
                                     int n,
                                     const T* __restrict__ w,
                                     const T* __restrict__ v,
                                     const T* __restrict__ rhs,
                                     T* __restrict__ S_lower,
                                     T* __restrict__ S_main,
                                     T* __restrict__ S_upper,
                                     T* __restrict__ S_rhs)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int s_size = 2 * m_pad / BLOCKDIM;

    if(gid < s_size)
    {
        S_upper[gid] = (gid % 2 == 0) ? v[gid / 2] : static_cast<T>(1);
        S_lower[gid]
            = (gid % 2 == 0) ? static_cast<T>(1) : w[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
    }

    if(gid == 0)
    {
        S_lower[0]          = static_cast<T>(0);
        S_lower[1]          = static_cast<T>(0);
        S_upper[s_size - 2] = static_cast<T>(0);
        S_upper[s_size - 1] = static_cast<T>(0);
        S_main[0]           = static_cast<T>(1);
        S_main[s_size - 1]  = static_cast<T>(1);
    }

    if(gid >= 1 && gid < s_size - 1)
    {
        S_main[gid]
            = (gid % 2 == 0) ? w[gid / 2] : v[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
    }

    if(gid < s_size / 2)
    {
        for(int batch = blockIdx.y; batch < n; batch += 32768)
        {
            S_rhs[2 * gid + s_size * batch] = rhs[gid + m_pad * batch];
            S_rhs[2 * gid + 1 + s_size * batch]
                = rhs[gid + (m_pad / BLOCKDIM) * (BLOCKDIM - 1) + m_pad * batch];
        }
    }
}

template <uint32_t S_SIZE, typename T>
__global__ void S_solve_kernel(int m,
                               int n,
                               const T* __restrict__ S_lower,
                               const T* __restrict__ S_main,
                               const T* __restrict__ S_upper,
                               T* __restrict__ rhs)
{
    static_assert(S_SIZE >= 2);

    const int batch = blockIdx.x;

    T mt[S_SIZE];

    constexpr int               PIVOT_MASK_WORDS = (S_SIZE + 31) / 32;
    PivotMask<PIVOT_MASK_WORDS> pivot_mask;

    int k  = 0;
    T   bk = S_main[k];

    while(k < S_SIZE)
    {
        T ck   = S_upper[k];
        T ck_1 = (k < (S_SIZE - 1)) ? S_upper[k + 1] : static_cast<T>(0);
        T bk_1 = (k < (S_SIZE - 1)) ? S_main[k + 1] : static_cast<T>(0);
        T ak_1 = (k < (S_SIZE - 1)) ? S_lower[k + 1] : static_cast<T>(0);
        T ak_2 = (k < (S_SIZE - 2)) ? S_lower[k + 2] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (S_SIZE - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            mt[k] = ck * inv_bk;

            pivot_mask.set_pivoting_to_1x1(k);

            // L * B * x = y
            T rhsk = rhs[k + m * batch] * inv_bk;

            rhs[k + m * batch] = rhsk;

            if(k < (S_SIZE - 1))
            {
                rhs[k + 1 + m * batch] += -(ak_1 * rhsk);

                bk_1 = bk_1 - ak_1 * ck * inv_bk;
            }

            bk = bk_1;

            k += 1;
        }
        else
        {
            const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

            mt[k] = -ck * ck_1 * det;

            pivot_mask.set_pivoting_to2x2(k);

            if(k < (S_SIZE - 1))
            {
                mt[k + 1] = bk * ck_1 * det;

                pivot_mask.set_pivoting_to2x2(k + 1);
            }

            T bk_2 = static_cast<T>(0);

            // L * B * x = y
            T rhsk   = rhs[k + m * batch] * det;
            T rhsk_1 = rhs[k + 1 + m * batch] * det;

            rhs[k + m * batch]     = (bk_1 * rhsk - ck * rhsk_1);
            rhs[k + 1 + m * batch] = (-ak_1 * rhsk + bk * rhsk_1);

            if(k < (S_SIZE - 2))
            {
                rhs[k + 2 + m * batch] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                bk_2 = S_main[k + 2];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }

    assert(k == S_SIZE);
    // at this point k = S_SIZE. Could just set k = S_SIZE - 1 here
    k--;

    k -= pivot_mask.get_pivoting(k);

    // backward solve (M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot_mask.get_pivoting(k) == 1)
        {
            const T tmp = mt[k];

            rhs[k + m * batch] += -tmp * rhs[k + 1 + m * batch];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[k];
            const T tmp2 = mt[k - 1];

            rhs[k + m * batch] += -tmp1 * rhs[k + 1 + m * batch];
            rhs[k - 1 + m * batch] += -tmp2 * rhs[k + 1 + m * batch];

            k -= 2;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void backward_solve_kernel(
    int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ rhs)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int nblocks = m_pad / BLOCKDIM;

    if(gid >= nblocks)
    {
        return;
    }

    for(int batch = blockIdx.y; batch < n; batch += 32768)
    {
        // backward solve (S * x = B_pad)
        double x1 = (gid >= 1) ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (gid - 1) + m_pad * batch] : 0.0f;
        double x2 = (gid < (m_pad / BLOCKDIM - 1)) ? rhs[gid + 1 + m_pad * batch] : 0.0f;

        for(int j = 1; j < BLOCKDIM - 1; j++)
        {
            rhs[(m_pad / BLOCKDIM) * j + gid + m_pad * batch]
                = rhs[(m_pad / BLOCKDIM) * j + gid + m_pad * batch]
                - w[(m_pad / BLOCKDIM) * j + gid] * x1 - v[(m_pad / BLOCKDIM) * j + gid] * x2;
        }
    }
}

// template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
// __global__ void data_marshaling_kernel(int m,
//                                        int m_pad,
//                                        const T* __restrict__ lower_in,
//                                        const T* __restrict__ main_in,
//                                        const T* __restrict__ upper_in,
//                                        T* __restrict__ lower_out,
//                                        T* __restrict__ main_out,
//                                        T* __restrict__ upper_out)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int gid = tid + BLOCKSIZE * bid;

//     // const int wid = tid / (BLOCKSIZE / BLOCKDIM); // 0, 1, 2,...7
//     // const int lid = tid % (BLOCKSIZE / BLOCKDIM); // 0, 1, 2,...31
//     const int wid = tid / (m_pad / BLOCKDIM); // 0, 1, 2,...7
//     const int lid = tid % (m_pad / BLOCKDIM); // 0, 1, 2,...31

//     __shared__ T tile[BLOCKSIZE];

//     // const int out_index = (BLOCKSIZE / BLOCKDIM) * (gridDim.x * wid + bid) + lid;
//     const int out_index = (m_pad / BLOCKDIM) * (gridDim.x * wid + bid) + lid;

//     tile[tid] = (gid < m) ? lower_in[gid] : static_cast<T>(0);
//     __syncthreads();

//     if(gid < m_pad)
//     {
//         lower_out[out_index] = tile[BLOCKDIM * lid + wid];
//     }
//     __syncthreads();

//     tile[tid] = (gid < m) ? main_in[gid] : static_cast<T>(1);
//     __syncthreads();

//     if(gid < m_pad)
//     {
//         main_out[out_index] = tile[BLOCKDIM * lid + wid];
//     }
//     __syncthreads();

//     tile[tid] = (gid < m) ? upper_in[gid] : static_cast<T>(0);
//     __syncthreads();

//     if(gid < m_pad)
//     {
//         upper_out[out_index] = tile[BLOCKDIM * lid + wid];
//     }
//     __syncthreads();
// }

// template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
// __global__ void LBM_wv_kernel(int m_pad,
//                         int n,
//                         int ldb,
//                         const T* __restrict__ a,
//                         const T* __restrict__ b,
//                         const T* __restrict__ c,
//                         T* __restrict__ w,
//                         T* __restrict__ v,
//                         T* __restrict__ mt,
//                         int* __restrict__ pivot)
// {
//     // From Bunch-Kaufman pivoting criteria
//     const double kappa = double(0.5) * (sqrt(double(5.0)) - double(1.0));

//     int tidx = threadIdx.x;
//     int bidx = blockIdx.x;
//     int gid  = tidx + BLOCKSIZE * bidx;

//     int nblocks = m_pad / BLOCKDIM;

//     if(gid >= nblocks)
//     {
//         return;
//     }

//     T bk                              = b[gid];
//     w[gid]                            = a[gid];
//     v[gid + (BLOCKDIM - 1) * nblocks] = c[gid + (BLOCKDIM - 1) * nblocks];

//     // forward solve (L* B * w = w and L* B * v = v)
//     int k = 0;
//     while(k < m_pad)
//     {
//         T ck   = c[k + gid];
//         T ak_1 = (k < (BLOCKDIM - 1) * nblocks) ? a[k + nblocks + gid] : static_cast<T>(0);
//         T bk_1 = (k < (BLOCKDIM - 1) * nblocks) ? b[k + nblocks + gid] : static_cast<T>(0);
//         T ck_1 = (k < (BLOCKDIM - 1) * nblocks) ? c[k + nblocks + gid] : static_cast<T>(0);
//         T ak_2 = (k < (BLOCKDIM - 2) * nblocks) ? a[k + 2 * nblocks + gid] : static_cast<T>(0);

//         // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
//         // pivoting criteria
//         double sigma = double(0);
//         sigma = max(double(abs(ak_1)), double(abs(ak_2)));
//         sigma = max(double(abs(bk_1)), sigma);
//         sigma = max(double(abs(ck)), sigma);
//         sigma = max(double(abs(ck_1)), sigma);

//         // 1x1 pivoting
//         if(abs(bk) * sigma >= kappa * abs(ak_1 * ck)
//             || k == (BLOCKDIM - 1) * nblocks)
//         {
//             T iBk = static_cast<T>(1) / bk;

//             bk_1 = bk_1 - ak_1 * ck * iBk;
//             ak_1 = ak_1 * iBk;
//             ck   = ck * iBk;

//             T wk = w[k + gid];
//             T vk = v[k + gid];

//             w[k + gid]     = wk * iBk;
//             v[k + gid]     = vk * iBk;
//             mt[k + gid]    = ck;
//             pivot[k + gid] = 1;

//             if(k < (BLOCKDIM - 1) * nblocks)
//             {
//                 w[k + nblocks + gid] += -ak_1 * wk;
//             }

//             bk = bk_1;
//             k += nblocks;
//         }
//         // 2x2 pivoting
//         else
//         {
//             T det = bk * bk_1 - ak_1 * ck;
//             det   = static_cast<T>(1) / det;

//             T wk   = w[k + gid];
//             T wk_1 = w[k + nblocks + gid];
//             T vk   = v[k + gid];
//             T vk_1 = v[k + nblocks + gid];

//             w[k + gid]     = (bk_1 * wk - ck * wk_1) * det;
//             v[k + gid]     = (bk_1 * vk - ck * vk_1) * det;
//             mt[k + gid]    = -ck * ck_1 * det;
//             pivot[k + gid] = 2;

//             if(k < (BLOCKDIM - 1) * nblocks)
//             {
//                 w[k + nblocks + gid]     = (-ak_1 * wk + bk * wk_1) * det;
//                 v[k + nblocks + gid]     = (-ak_1 * vk + bk * vk_1) * det;
//                 mt[k + nblocks + gid]    = bk * ck_1 * det;
//                 pivot[k + nblocks + gid] = 2;
//             }

//             T bk_2 = static_cast<T>(0);

//             if(k < (BLOCKDIM - 2) * nblocks)
//             {
//                 w[k + 2 * nblocks + gid]
//                     += -(-ak_1 * ak_2 * det) * wk - (bk * ak_2 * det) * wk_1;

//                 bk_2 = b[k + 2 * nblocks + gid];
//                 bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
//             }

//             bk = bk_2;
//             k += 2 * nblocks;
//         }
//     }

//     __threadfence();

//     // at this point k = BLOCKDIM * nblocks
//     k -= nblocks;

//     k -= nblocks * pivot[k + gid];

//     // backward solve (M^T * w = w and M^T * v = v)
//     while(k >= 0)
//     {
//         if(pivot[k + gid] == 1)
//         {
//             T tmp = mt[k + gid];
//             w[k + gid] += -tmp * w[k + nblocks + gid];
//             v[k + gid] += -tmp * v[k + nblocks + gid];

//             k -= nblocks;
//         }
//         else
//         {
//             T tmp1 = mt[k + gid];
//             T tmp2 = mt[k - nblocks + gid];

//             w[k + gid] += -tmp1 * w[k + nblocks + gid];
//             w[k - nblocks + gid] += -tmp2 * w[k + nblocks + gid];
//             v[k + gid] += -tmp1 * v[k + nblocks + gid];
//             v[k - nblocks + gid] += -tmp2 * v[k + nblocks + gid];

//             k -= 2 * nblocks;
//         }
//     }
// }

// Kernel that combines the swap-adjacent-pairs and scatter-to-B_pad steps that
// previously ran on the host.  Each thread i in [0, s_size/2) handles one pair
// of output elements, applying the same index permutation as the two host loops:
//
//   for(i = 1; i < s_size-1; i += 2)  swap(y[i], y[i+1]);
//   for(i = 0; i < s_size/2; i++)
//       B_pad[i]        = y[2*i];
//       B_pad[i+stride] = y[2*i+1];
//
// The swap is folded into the read so no intermediate storage is needed.
// Grid: dim3((s_size/2 + BLOCKSIZE-1) / BLOCKSIZE, n)
//
// for(int i = 1; i < s_size - 1; i += 2)
// {
//     double temp = h_y[i];
//     h_y[i]      = h_y[i + 1];
//     h_y[i + 1]  = temp;
// }
// for(int i = 0; i < s_size / 2; i++)
// {
//     h_B_pad[i]                                       = h_y[2 * i];
//     h_B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)] = h_y[2 * i + 1];
// }
template <uint32_t BLOCKDIM, uint32_t BLOCKSIZE, typename T>
__global__ void scatter_S_B_to_B_pad_kernel(int s_size,
                                          int m_pad,
                                          int n,
                                          const T* __restrict__ S_B,
                                          T* __restrict__ B_pad)
{
    const int i     = blockIdx.x * BLOCKSIZE + threadIdx.x; // [0, s_size/2)

    if(i >= s_size / 2)
        return;

    for(int batch = blockIdx.y; batch < n; batch += 32768)
    {
        const T* S_B_j   = S_B + batch * s_size;
        T*       B_pad_j = B_pad + batch * m_pad;

        const int stride = (m_pad / BLOCKDIM) * (BLOCKDIM - 1);

        // After the swap loop, element at position 2*i is:
        //   i == 0  -> S_B[0]       (not touched by the swap)
        //   i  > 0  -> S_B[2*i - 1] (position 2*i was swapped with 2*i-1)
        const T val_even = (i == 0) ? S_B_j[0] : S_B_j[2 * i - 1];

        // After the swap loop, element at position 2*i+1 is:
        //   2*i+1 < s_size-1  -> S_B[2*i + 2] (swapped with its right neighbour)
        //   2*i+1 == s_size-1 -> S_B[s_size-1] (last element, not touched)
        const T val_odd = (2 * i + 1 < s_size - 1) ? S_B_j[2 * i + 2] : S_B_j[s_size - 1];

        B_pad_j[i]          = val_even;
        B_pad_j[i + stride] = val_odd;
    }
}

#endif // TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H
