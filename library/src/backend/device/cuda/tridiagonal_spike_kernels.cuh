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

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
__global__ void data_marshaling_kernel(int m_in,
                                        int m_out,
                                        const T* __restrict__ lower_in,
                                        const T* __restrict__ main_in,
                                        const T* __restrict__ upper_in,
                                        T* __restrict__ lower_out,
                                        T* __restrict__ main_out,
                                        T* __restrict__ upper_out)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int wid = tid / WF_SIZE;
    const int lid = tid % WF_SIZE;

    __shared__ T tile1[BLOCKSIZE];
    __shared__ T tile2[BLOCKSIZE];

    tile1[tid] = (gid < m_in) ? lower_in[gid] : static_cast<T>(1);
    __syncthreads();

    tile2[WF_SIZE * wid + lid] = tile1[WF_SIZE * lid + wid];
    __syncthreads();

    if(gid < m_out)
    {
        lower_out[gid] = tile2[tid];
    }

    tile1[tid] = (gid < m_in) ? main_in[gid] : static_cast<T>(1);
    __syncthreads();

    tile2[WF_SIZE * wid + lid] = tile1[WF_SIZE * lid + wid];
    __syncthreads();

    if(gid < m_out)
    {
        main_out[gid] = tile2[tid];
    }

    tile1[tid] = (gid < m_in) ? upper_in[gid] : static_cast<T>(1);
    __syncthreads();

    tile2[WF_SIZE * wid + lid] = tile1[WF_SIZE * lid + wid];
    __syncthreads();

    if(gid < m_out)
    {
        upper_out[gid] = tile2[tid];
    }
}














    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    __global__ void LBM_wv_kernel(int m_pad,
                            int n,
                            int ldb,
                            const T* __restrict__ a,
                            const T* __restrict__ b,
                            const T* __restrict__ c,
                            T* __restrict__ w,
                            T* __restrict__ v,
                            T* __restrict__ mt,
                            int* __restrict__ pivot)
    {
        // From Bunch-Kaufman pivoting criteria
        const double kappa = double(0.5) * (sqrt(double(5.0)) - double(1.0));

        int tidx = threadIdx.x;
        int bidx = blockIdx.x;
        int gid  = tidx + BLOCKSIZE * bidx;

        int nblocks = m_pad / BLOCKDIM;

        if(gid >= nblocks)
        {
            return;
        }

        T bk                              = b[gid];
        w[gid]                            = a[gid];
        v[gid + (BLOCKDIM - 1) * nblocks] = c[gid + (BLOCKDIM - 1) * nblocks];

        // forward solve (L* B * w = w and L* B * v = v)
        int k = 0;
        while(k < m_pad)
        {
            T ck   = c[k + gid];
            T ak_1 = (k < (BLOCKDIM - 1) * nblocks) ? a[k + nblocks + gid] : static_cast<T>(0);
            T bk_1 = (k < (BLOCKDIM - 1) * nblocks) ? b[k + nblocks + gid] : static_cast<T>(0);
            T ck_1 = (k < (BLOCKDIM - 1) * nblocks) ? c[k + nblocks + gid] : static_cast<T>(0);
            T ak_2 = (k < (BLOCKDIM - 2) * nblocks) ? a[k + 2 * nblocks + gid] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            double sigma = double(0);
            sigma = max(double(abs(ak_1)), double(abs(ak_2)));
            sigma = max(double(abs(bk_1)), sigma);
            sigma = max(double(abs(ck)), sigma);
            sigma = max(double(abs(ck_1)), sigma);

            // 1x1 pivoting
            if(abs(bk) * sigma >= kappa * abs(ak_1 * ck)
               || k == (BLOCKDIM - 1) * nblocks)
            {
                T iBk = static_cast<T>(1) / bk;

                bk_1 = bk_1 - ak_1 * ck * iBk;
                ak_1 = ak_1 * iBk;
                ck   = ck * iBk;

                T wk = w[k + gid];
                T vk = v[k + gid];

                w[k + gid]     = wk * iBk;
                v[k + gid]     = vk * iBk;
                mt[k + gid]    = ck;
                pivot[k + gid] = 1;

                if(k < (BLOCKDIM - 1) * nblocks)
                {
                    w[k + nblocks + gid] += -ak_1 * wk;
                }

                bk = bk_1;
                k += nblocks;
            }
            // 2x2 pivoting
            else
            {
                T det = bk * bk_1 - ak_1 * ck;
                det   = static_cast<T>(1) / det;

                T wk   = w[k + gid];
                T wk_1 = w[k + nblocks + gid];
                T vk   = v[k + gid];
                T vk_1 = v[k + nblocks + gid];

                w[k + gid]     = (bk_1 * wk - ck * wk_1) * det;
                v[k + gid]     = (bk_1 * vk - ck * vk_1) * det;
                mt[k + gid]    = -ck * ck_1 * det;
                pivot[k + gid] = 2;

                if(k < (BLOCKDIM - 1) * nblocks)
                {
                    w[k + nblocks + gid]     = (-ak_1 * wk + bk * wk_1) * det;
                    v[k + nblocks + gid]     = (-ak_1 * vk + bk * vk_1) * det;
                    mt[k + nblocks + gid]    = bk * ck_1 * det;
                    pivot[k + nblocks + gid] = 2;
                }

                T bk_2 = static_cast<T>(0);

                if(k < (BLOCKDIM - 2) * nblocks)
                {
                    w[k + 2 * nblocks + gid]
                        += -(-ak_1 * ak_2 * det) * wk - (bk * ak_2 * det) * wk_1;

                    bk_2 = b[k + 2 * nblocks + gid];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2 * nblocks;
            }
        }

        __threadfence();

        // at this point k = BLOCKDIM * nblocks
        k -= nblocks;

        k -= nblocks * pivot[k + gid];

        // backward solve (M^T * w = w and M^T * v = v)
        while(k >= 0)
        {
            if(pivot[k + gid] == 1)
            {
                T tmp = mt[k + gid];
                w[k + gid] += -tmp * w[k + nblocks + gid];
                v[k + gid] += -tmp * v[k + nblocks + gid];

                k -= nblocks;
            }
            else
            {
                T tmp1 = mt[k + gid];
                T tmp2 = mt[k - nblocks + gid];

                w[k + gid] += -tmp1 * w[k + nblocks + gid];
                w[k - nblocks + gid] += -tmp2 * w[k + nblocks + gid];
                v[k + gid] += -tmp1 * v[k + nblocks + gid];
                v[k - nblocks + gid] += -tmp2 * v[k + nblocks + gid];

                k -= 2 * nblocks;
            }
        }
    }






#endif // TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H
