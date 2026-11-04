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

#ifndef TRIDIAGONAL_SPIKE_FUSED_KERNELS_H
#define TRIDIAGONAL_SPIKE_FUSED_KERNELS_H

#include <assert.h>

#include "common.cuh"
#include "pivot_mask.cuh"

// Sets bit k to 0 to record a 1x1 pivot at row k.
__device__ __forceinline__ void set_pivoting_to_1x1(int* mask, int k)
{
    *mask &= ~(1u << k);
}

// Sets bit k to 1 to record 2x2 pivoting at row k.
__device__ __forceinline__ void set_pivoting_to_2x2(int* mask, int k)
{
    *mask |= (1u << k);
}

// Returns 1 if row k used 1x1 pivoting, 2 if row k is part of a 2x2 pivot.
__device__ __forceinline__ int get_pivoting(int mask, int k)
{
    return ((mask >> k) & 1u) ? 2 : 1;
}

// Need to store:
// bk_1 * det
// ck * det
// ak_2
// -ak_1 * det
// bk * det
// mt
// w
// v
template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void solve_fused_wvmt_kernel(int m,
                                        const T* __restrict__ lower,
                                        const T* __restrict__ main,
                                        const T* __restrict__ upper,
                                        T* __restrict__ coeff1,
                                        T* __restrict__ coeff2,
                                        T* __restrict__ coeff3,
                                        T* __restrict__ coeff4,
                                        T* __restrict__ coeff5,
                                        T* __restrict__ w,
                                        T* __restrict__ v,
                                        T* __restrict__ mt,
                                        int* __restrict__ pivot_mask,
                                        T* __restrict__ s_coeff1,
                                        T* __restrict__ s_coeff2,
                                        T* __restrict__ s_coeff3,
                                        T* __restrict__ s_coeff4,
                                        T* __restrict__ s_coeff5,
                                        T* __restrict__ s_mt,
                                        int* __restrict__ s_pivot_mask)
{
    static_assert(BLOCKDIM >= 2);
    static_assert(BLOCKDIM % 2 == 0);
    static_assert(BLOCKSIZE % 2 == 0);

    const int tid = threadIdx.x;

    __shared__ T slower[BLOCKSIZE * BLOCKDIM];
    __shared__ T smain[BLOCKSIZE * BLOCKDIM];
    __shared__ T supper[BLOCKSIZE * BLOCKDIM];

    for(int i = tid; i < BLOCKSIZE * BLOCKDIM; i += BLOCKSIZE)
    {
        slower[i] = (i < m) ? lower[i] : static_cast<T>(0);
        smain[i]  = (i < m) ? main[i] : static_cast<T>(1);
        supper[i] = (i < m) ? upper[i] : static_cast<T>(0);
    }

    __syncthreads();

    // LBMT solve
    T bk = smain[BLOCKDIM * tid];

    w[tid]                              = slower[BLOCKDIM * tid];
    v[BLOCKSIZE * (BLOCKDIM - 1) + tid] = supper[BLOCKDIM * tid + (BLOCKDIM - 1)];

    int k = 0;
    while(k < BLOCKDIM)
    {
        T ck   = supper[BLOCKDIM * tid + k];
        T ck_1 = (k < (BLOCKDIM - 1)) ? supper[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
        T bk_1 = (k < (BLOCKDIM - 1)) ? smain[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
        T ak_1 = (k < (BLOCKDIM - 1)) ? slower[BLOCKDIM * tid + (k + 1)] : static_cast<T>(0);
        T ak_2 = (k < (BLOCKDIM - 2)) ? slower[BLOCKDIM * tid + (k + 2)] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (BLOCKDIM - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            const T wk = w[BLOCKSIZE * k + tid];
            const T vk = v[BLOCKSIZE * k + tid];

            w[BLOCKSIZE * k + tid]  = wk * inv_bk;
            v[BLOCKSIZE * k + tid]  = vk * inv_bk;
            mt[BLOCKSIZE * k + tid] = ck * inv_bk;

            set_pivoting_to_1x1(&pivot_mask[tid], k);

            if(k < (BLOCKDIM - 1))
            {
                w[BLOCKSIZE * (k + 1) + tid] += -ak_1 * wk * inv_bk;
            }

            coeff1[BLOCKSIZE * k + tid] = inv_bk;
            coeff2[BLOCKSIZE * k + tid] = ak_1;

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

            const T wk   = w[BLOCKSIZE * k + tid];
            const T wk_1 = w[BLOCKSIZE * (k + 1) + tid];
            const T vk   = v[BLOCKSIZE * k + tid];
            const T vk_1 = v[BLOCKSIZE * (k + 1) + tid];

            w[BLOCKSIZE * k + tid]  = (bk_1 * wk - ck * wk_1) * det;
            v[BLOCKSIZE * k + tid]  = (bk_1 * vk - ck * vk_1) * det;
            mt[BLOCKSIZE * k + tid] = -ck * ck_1 * det;

            set_pivoting_to_2x2(&pivot_mask[tid], k);

            if(k < (BLOCKDIM - 1))
            {
                w[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * wk + bk * wk_1) * det;
                v[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * vk + bk * vk_1) * det;
                mt[BLOCKSIZE * (k + 1) + tid] = bk * ck_1 * det;

                set_pivoting_to_2x2(&pivot_mask[tid], k + 1);
            }

            T bk_2 = static_cast<T>(0);

            if(k < (BLOCKDIM - 2))
            {
                w[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
            }

            // |bk   ck  ||xk  |   |rhsk   |
            // |ak_1 bk_1||xk_1| = |rhsk _1|
            //
            //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
            //                                  |-ak_1  bk |
            coeff1[BLOCKSIZE * k + tid] = bk_1 * det;
            coeff2[BLOCKSIZE * k + tid] = ck * det;
            coeff3[BLOCKSIZE * k + tid] = ak_2;
            coeff4[BLOCKSIZE * k + tid] = -ak_1 * det;
            coeff5[BLOCKSIZE * k + tid] = bk * det;

            if(k < (BLOCKDIM - 2))
            {
                bk_2 = smain[BLOCKDIM * tid + k + 2];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= get_pivoting(pivot_mask[tid], k);

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(get_pivoting(pivot_mask[tid], k) == 1)
        {
            const T tmp = mt[BLOCKSIZE * k + tid];

            w[BLOCKSIZE * k + tid] += -tmp * w[BLOCKSIZE * (k + 1) + tid];
            v[BLOCKSIZE * k + tid] += -tmp * v[BLOCKSIZE * (k + 1) + tid];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[BLOCKSIZE * k + tid];
            const T tmp2 = mt[BLOCKSIZE * (k - 1) + tid];

            w[BLOCKSIZE * k + tid] += -tmp1 * w[BLOCKSIZE * (k + 1) + tid];
            w[BLOCKSIZE * (k - 1) + tid] += -tmp2 * w[BLOCKSIZE * (k + 1) + tid];
            v[BLOCKSIZE * k + tid] += -tmp1 * v[BLOCKSIZE * (k + 1) + tid];
            v[BLOCKSIZE * (k - 1) + tid] += -tmp2 * v[BLOCKSIZE * (k + 1) + tid];

            k -= 2;
        }
    }

    // Fill S system
    constexpr int S_SIZE = 2 * BLOCKSIZE;

    __shared__ T S_lower[S_SIZE];
    __shared__ T S_main[S_SIZE];
    __shared__ T S_upper[S_SIZE];

    S_lower[2 * tid]     = static_cast<T>(1);
    S_lower[2 * tid + 1] = w[BLOCKSIZE * (BLOCKDIM - 1) + tid];

    S_upper[2 * tid]     = v[tid];
    S_upper[2 * tid + 1] = static_cast<T>(1);

    S_main[2 * tid]     = w[tid];
    S_main[2 * tid + 1] = v[BLOCKSIZE * (BLOCKDIM - 1) + tid];

    if(tid == 0)
    {
        S_lower[0] = static_cast<T>(0);
        S_lower[1] = static_cast<T>(0);

        S_main[0] = static_cast<T>(1);
    }

    if(tid == (BLOCKSIZE - 1))
    {
        S_upper[2 * (BLOCKSIZE - 1)]     = static_cast<T>(0);
        S_upper[2 * (BLOCKSIZE - 1) + 1] = static_cast<T>(0);

        S_main[2 * (BLOCKSIZE - 1) + 1] = static_cast<T>(1);
    }

    __syncthreads();

    // Solve S system (solved by 1 thread)
    if(tid == 0)
    {
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

                s_mt[k] = ck * inv_bk;

                set_pivoting_to_1x1(&s_pivot_mask[k / 32], (k % 32));

                s_coeff1[k] = inv_bk;
                s_coeff2[k] = ak_1;

                if(k < (S_SIZE - 1))
                {
                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                s_mt[k] = -ck * ck_1 * det;

                set_pivoting_to_2x2(&s_pivot_mask[k / 32], (k % 32));

                if(k < (S_SIZE - 1))
                {
                    s_mt[k + 1] = bk * ck_1 * det;

                    set_pivoting_to_2x2(&s_pivot_mask[(k + 1) / 32], ((k + 1) % 32));
                }

                T bk_2 = static_cast<T>(0);

                s_coeff1[k] = bk_1 * det;
                s_coeff2[k] = ck * det;
                s_coeff3[k] = ak_2;
                s_coeff4[k] = -ak_1 * det;
                s_coeff5[k] = bk * det;

                if(k < (S_SIZE - 2))
                {
                    bk_2 = S_main[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
__global__ void solve_fused_rhs_kernel(int m,
                                   const T* __restrict__ coeff1,
                                   const T* __restrict__ coeff2,
                                   const T* __restrict__ coeff3,
                                   const T* __restrict__ coeff4,
                                   const T* __restrict__ coeff5,
                                   const T* __restrict__ w,
                                   const T* __restrict__ v,
                                   const T* __restrict__ mt,
                                   const int* mask,
                                   const T* __restrict__ s_coeff1,
                                   const T* __restrict__ s_coeff2,
                                   const T* __restrict__ s_coeff3,
                                   const T* __restrict__ s_coeff4,
                                   const T* __restrict__ s_coeff5,
                                   const T* __restrict__ s_mt,
                                   const int* s_mask,
                                   const T* __restrict__ B,
                                   T* __restrict__ X)
{
    static_assert(BLOCKDIM >= 2);
    static_assert(BLOCKDIM <= 32);
    static_assert(BLOCKDIM % 2 == 0);
    static_assert(BLOCKSIZE % 2 == 0);
    static_assert(NCOLS <= BLOCKSIZE);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    T rhs_reg[BLOCKDIM * NCOLS];
    for(int c = 0; c < NCOLS; c++)
    {
        for(int i = 0; i < BLOCKDIM; i++)
        {
            rhs_reg[BLOCKDIM * c + i] = (BLOCKDIM * tid + i < m)
                                            ? B[BLOCKDIM * tid + i + m * (NCOLS * bid + c)]
                                            : static_cast<T>(0);
        }
    }

    // LBMT solve
    int k = 0;
    while(k < BLOCKDIM)
    {
        // 1x1 pivoting
        if(get_pivoting(mask[tid], k) == 1 || k == (BLOCKDIM - 1))
        {
            const T inv_bk = coeff1[BLOCKSIZE * k + tid];
            const T ak_1 = coeff2[BLOCKSIZE * k + tid];

            // L * B * x = y
            for(int c = 0; c < NCOLS; c++)
            {
                const T rhsk = rhs_reg[BLOCKDIM * c + k] * inv_bk;

                rhs_reg[BLOCKDIM * c + k] = rhsk;

                if(k < (BLOCKDIM - 1))
                {
                    rhs_reg[BLOCKDIM * c + (k + 1)] += -(ak_1 * rhsk);
                }
            }

            k += 1;
        }
        else
        {
            // |bk   ck  ||xk  |   |rhsk   |
            // |ak_1 bk_1||xk_1| = |rhsk _1|
            //
            //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
            //                                  |-ak_1  bk |

            const T bk_1_det = coeff1[BLOCKSIZE * k + tid];
            const T ck_det = coeff2[BLOCKSIZE * k + tid];
            const T ak_2 = coeff3[BLOCKSIZE * k + tid];
            const T nak_1_det = coeff4[BLOCKSIZE * k + tid];
            const T bk_det = coeff5[BLOCKSIZE * k + tid];

            // L * B * x = y
            for(int c = 0; c < NCOLS; c++)
            {
                const T rhsk   = rhs_reg[BLOCKDIM * c + k];
                const T rhsk_1 = rhs_reg[BLOCKDIM * c + k + 1];

                rhs_reg[BLOCKDIM * c + k]     = (bk_1_det * rhsk - ck_det * rhsk_1);
                rhs_reg[BLOCKDIM * c + k + 1] = (nak_1_det * rhsk + bk_det * rhsk_1);

                if(k < (BLOCKDIM - 2))
                {
                    rhs_reg[BLOCKDIM * c + k + 2]
                        += -(nak_1_det * ak_2 * rhsk + ak_2 * bk_det * rhsk_1);
                }
            }

            k += 2;
        }
    }

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= get_pivoting(mask[tid], k);

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(get_pivoting(mask[tid], k) == 1)
        {
            const T tmp = mt[BLOCKSIZE * k + tid];

            for(int c = 0; c < NCOLS; c++)
            {
                rhs_reg[BLOCKDIM * c + k] += -tmp * rhs_reg[BLOCKDIM * c + k + 1];
            }

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[BLOCKSIZE * k + tid];
            const T tmp2 = mt[BLOCKSIZE * (k - 1) + tid];

            for(int c = 0; c < NCOLS; c++)
            {
                rhs_reg[BLOCKDIM * c + k] += -tmp1 * rhs_reg[BLOCKDIM * c + k + 1];
                rhs_reg[BLOCKDIM * c + k - 1] += -tmp2 * rhs_reg[BLOCKDIM * c + k + 1];
            }

            k -= 2;
        }
    }

    // Fill S system
    constexpr int S_SIZE = 2 * BLOCKSIZE;

    __shared__ T S_rhs[S_SIZE * NCOLS];

    for(int c = 0; c < NCOLS; c++)
    {
        S_rhs[S_SIZE * c + 2 * tid]     = rhs_reg[BLOCKDIM * c + 0];
        S_rhs[S_SIZE * c + 2 * tid + 1] = rhs_reg[BLOCKDIM * c + (BLOCKDIM - 1)];
    }

    __syncthreads();

    // Solve S system (solved by 1 thread)
    if(tid < NCOLS)
    {
        int c = tid;
        int k  = 0;

        while(k < S_SIZE)
        {
            // 1x1 pivoting
            if(get_pivoting(s_mask[k / 32], (k % 32)) == 1 || k == (S_SIZE - 1))
            {
                const T inv_bk = s_coeff1[k];
                const T ak_1 = s_coeff2[k];

                // L * B * x = y
                const T rhsk = S_rhs[S_SIZE * c + k] * inv_bk;

                S_rhs[S_SIZE * c + k] = rhsk;

                if(k < (S_SIZE - 1))
                {
                    S_rhs[S_SIZE * c + k + 1] += -(ak_1 * rhsk);
                }

                k += 1;
            }
            else
            {
                const T bk_1_det = s_coeff1[k];
                const T ck_det = s_coeff2[k];
                const T ak_2 = s_coeff3[k];
                const T nak_1_det = s_coeff4[k];
                const T bk_det = s_coeff5[k];

                // L * B * x = y
                const T rhsk   = S_rhs[S_SIZE * c + k];
                const T rhsk_1 = S_rhs[S_SIZE * c + k + 1];

                S_rhs[S_SIZE * c + k]     = (bk_1_det * rhsk - ck_det * rhsk_1);
                S_rhs[S_SIZE * c + k + 1] = (nak_1_det * rhsk + bk_det * rhsk_1);

                if(k < (S_SIZE - 2))
                {
                    S_rhs[S_SIZE * c + k + 2] += -(nak_1_det * ak_2 * rhsk + ak_2 * bk_det * rhsk_1);
                }

                k += 2;
            }
        }

        assert(k == S_SIZE);
        // at this point k = S_SIZE. Could just set k = S_SIZE - 1 here
        k--;

        k -= get_pivoting(s_mask[k / 32], (k % 32));

        // backward solve (M^T * rhs = rhs)
        while(k >= 0)
        {
            // if(pivot_mask.get_pivoting(k) == 1)
            if(get_pivoting(s_mask[k / 32], (k % 32)) == 1)
            {
                const T tmp = s_mt[k];

                S_rhs[S_SIZE * c + k] += -tmp * S_rhs[S_SIZE * c + k + 1];

                k -= 1;
            }
            else
            {
                const T tmp1 = s_mt[k];
                const T tmp2 = s_mt[k - 1];

                S_rhs[S_SIZE * c + k] += -tmp1 * S_rhs[S_SIZE * c + k + 1];
                S_rhs[S_SIZE * c + k - 1] += -tmp2 * S_rhs[S_SIZE * c + k + 1];

                k -= 2;
            }
        }
    }

    __syncthreads();

    // Scatter
    T x1[NCOLS];
    T x2[NCOLS];
    for(int c = 0; c < NCOLS; c++)
    {
        // After the swap loop, element at position 2*tid is:
        //   tid == 0  -> S_rhs[0]       (not touched by the swap)
        //   tid  > 0  -> S_rhs[2*tid - 1] (position 2*tid was swapped with 2*tid-1)
        const T val_even = (tid == 0) ? S_rhs[S_SIZE * c + 0] : S_rhs[S_SIZE * c + 2 * tid - 1];

        // After the swap loop, element at position 2*i+1 is:
        //   2*tid+1 < S_SIZE-1  -> S_rhs[2*tid + 2] (swapped with its right neighbour)
        //   2*tid+1 == S_SIZE-1 -> S_rhs[S_SIZE-1] (last element, not touched)
        const T val_odd = (2 * tid + 1 < S_SIZE - 1) ? S_rhs[S_SIZE * c + 2 * tid + 2]
                                                     : S_rhs[S_SIZE * c + S_SIZE - 1];

        rhs_reg[BLOCKDIM * c + 0]            = val_even;
        rhs_reg[BLOCKDIM * c + BLOCKDIM - 1] = val_odd;

        // backward solve (S * x = rhs)
        const T tmp1
            = __shfl_sync(0xFFFFFFFF, rhs_reg[BLOCKDIM * c + (BLOCKDIM - 1)], tid - 1, BLOCKSIZE);
        const T tmp2 = __shfl_sync(0xFFFFFFFF, rhs_reg[BLOCKDIM * c + 0], tid + 1, BLOCKSIZE);
        x1[c]   = (tid >= 1) ? tmp1 : static_cast<T>(0);
        x2[c]   = (tid < (BLOCKSIZE - 1)) ? tmp2 : static_cast<T>(0);
    }

    for(int j = 1; j < BLOCKDIM - 1; j++)
    {
        const T wj = w[BLOCKSIZE * j + tid];
        const T vj = v[BLOCKSIZE * j + tid];

        for(int c = 0; c < NCOLS; c++)
        {
            rhs_reg[BLOCKDIM * c + j] = rhs_reg[BLOCKDIM * c + j] - wj * x1[c] - vj * x2[c];
        }
    }

    // Transpose back
    for(int c = 0; c < NCOLS; c++)
    {
        for(int i = 0; i < BLOCKDIM; i++)
        {
            if((BLOCKDIM * tid + i) < m)
            {
                X[BLOCKDIM * tid + i + m * (NCOLS * bid + c)] = rhs_reg[BLOCKDIM * c + i];
            }
        }
    }
}

#endif // TRIDIAGONAL_SPIKE_FUSED_KERNELS_H
