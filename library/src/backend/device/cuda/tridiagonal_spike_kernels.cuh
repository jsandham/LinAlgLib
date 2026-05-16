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
                                       const T* __restrict__ lower_in,
                                       const T* __restrict__ main_in,
                                       const T* __restrict__ upper_in,
                                       const T* __restrict__ B_in,
                                       T* __restrict__ lower_out,
                                       T* __restrict__ main_out,
                                       T* __restrict__ upper_out,
                                       T* __restrict__ B_out)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int gwid = gid / (m_pad / BLOCKDIM);
    const int glid = gid % (m_pad / BLOCKDIM);

    lower_out[gid] = (gid < m) ? lower_in[BLOCKDIM * glid + gwid] : static_cast<T>(0);
    main_out[gid]  = (gid < m) ? main_in[BLOCKDIM * glid + gwid] : static_cast<T>(1);
    upper_out[gid] = (gid < m) ? upper_in[BLOCKDIM * glid + gwid] : static_cast<T>(0);
    B_out[gid]     = (gid < m) ? B_in[BLOCKDIM * glid + gwid] : static_cast<T>(0);
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void
    data_marshaling_kernel2(int m, int m_pad, const T* __restrict__ B_pad, T* __restrict__ B)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int gwid = gid / (m_pad / BLOCKDIM);
    const int glid = gid % (m_pad / BLOCKDIM);

    // B[gid]     = (gid < m) ? B_pad[BLOCKDIM * glid + gwid] : static_cast<T>(0);
    if(gid < m)
    {
        B[BLOCKDIM * glid + gwid] = B_pad[gid];
    }
}

template <typename T>
__device__ bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
{
    double kappa = double(0.5) * (sqrt(double(5.0)) - double(1.0));

    double sigma = double(0);
    sigma        = max(double(abs(ak_1)), double(abs(ak_2)));
    sigma        = max(double(abs(bk_1)), sigma);
    sigma        = max(double(abs(ck)), sigma);
    sigma        = max(double(abs(ck_1)), sigma);

    return abs(bk) * sigma >= kappa * abs(ak_1 * ck);
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
                                  T* __restrict__ rhs,
                                  int* __restrict__ pivot)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int nblocks = m_pad / BLOCKDIM;

    if(gid >= nblocks)
    {
        return;
    }

    T bk = main[gid];

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

            w[nblocks * k + gid]     = wk * inv_bk;
            v[nblocks * k + gid]     = vk * inv_bk;
            mt[nblocks * k + gid]    = ck * inv_bk;
            pivot[nblocks * k + gid] = 1;

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

            w[nblocks * k + gid]     = (bk_1 * wk - ck * wk_1) * det;
            v[nblocks * k + gid]     = (bk_1 * vk - ck * vk_1) * det;
            mt[nblocks * k + gid]    = -ck * ck_1 * det;
            pivot[nblocks * k + gid] = 2;

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid]     = (-ak_1 * wk + bk * wk_1) * det;
                v[nblocks * (k + 1) + gid]     = (-ak_1 * vk + bk * vk_1) * det;
                mt[nblocks * (k + 1) + gid]    = bk * ck_1 * det;
                pivot[nblocks * (k + 1) + gid] = 2;
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
                rhs[nblocks * (k + 2) + gid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                bk_2 = main[nblocks * (k + 2) + gid];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k += 2;
        }
    }
    __threadfence(); // I dont think I need this here since each thread is working on independent data.
    // I think this is only necessary if we have inter-thread dependencies, which we dont in this case.

    assert(k == BLOCKDIM);
    // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
    k--;

    k -= pivot[nblocks * k + gid];

    // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot[nblocks * k + gid] == 1)
        {
            const T tmp = mt[nblocks * k + gid];

            // I think k will always be less than BLOCKDIM - 1 here??
            w[nblocks * k + gid] += -tmp * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp * v[nblocks * (k + 1) + gid];
            rhs[nblocks * k + gid] += -tmp * rhs[nblocks * (k + 1) + gid];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[nblocks * k + gid];
            const T tmp2 = mt[nblocks * (k - 1) + gid];

            // I think k will always be less than BLOCKDIM - 2 here??
            w[nblocks * k + gid] += -tmp1 * w[nblocks * (k + 1) + gid];
            w[nblocks * (k - 1) + gid] += -tmp2 * w[nblocks * (k + 1) + gid];
            v[nblocks * k + gid] += -tmp1 * v[nblocks * (k + 1) + gid];
            v[nblocks * (k - 1) + gid] += -tmp2 * v[nblocks * (k + 1) + gid];
            rhs[nblocks * k + gid] += -tmp1 * rhs[nblocks * (k + 1) + gid];
            rhs[nblocks * (k - 1) + gid] += -tmp2 * rhs[nblocks * (k + 1) + gid];

            k -= 2;
        }
    }
}



// Complete Sx = B_pad
// for(int i = 0; i < m_pad / BLOCKDIM; i++)
// {
//     double x1 = (i >= 1) ? h_B_pad[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (i - 1)] : 0.0f;
//     double x2 = (i < (m_pad / BLOCKDIM - 1)) ? h_B_pad[i + 1] : 0.0f;

//     for(int j = 1; j < BLOCKDIM - 1; j++)
//     {
//         h_B_pad[(m_pad / BLOCKDIM) * j + i] = h_B_pad[(m_pad / BLOCKDIM) * j + i]
//                                             - h_w_pad[(m_pad / BLOCKDIM) * j + i] * x1
//                                             - h_v_pad[(m_pad / BLOCKDIM) * j + i] * x2;
//     }
// }


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

    // backward solve (S * x = B_pad)
    double x1 = (gid >= 1) ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (gid - 1)] : 0.0f;
    double x2 = (gid < (m_pad / BLOCKDIM - 1)) ? rhs[gid + 1] : 0.0f;

    for(int j = 1; j < BLOCKDIM - 1; j++)
    {
        rhs[(m_pad / BLOCKDIM) * j + gid] = rhs[(m_pad / BLOCKDIM) * j + gid]
                                            - w[(m_pad / BLOCKDIM) * j + gid] * x1
                                            - v[(m_pad / BLOCKDIM) * j + gid] * x2;
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

#endif // TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H
