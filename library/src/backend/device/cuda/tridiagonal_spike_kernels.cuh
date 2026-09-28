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

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
__device__ void
    data_transpose_device_gemini(int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out, T pad_value)
{
    // Each CUDA block handles PARTITIONS_PER_GROUP (e.g., 256) total partitions,
    // but processes them in smaller chunks dictated by BLOCKSIZE (e.g., 128).
    const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
    const int nblocks = m_pad / BLOCKDIM;

    // Prevent completely out-of-bounds blocks from executing
    if(group_start_partition >= nblocks)
        return;

    // Shared memory now only needs to hold BLOCKSIZE partitions at a time (~33 KB)
    __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

    const int tid = threadIdx.x;
    const int global_offset_out = group_start_partition * BLOCKDIM;

    // Loop over the group in chunks of BLOCKSIZE (e.g., 0, then 128)
    for (int chunk = 0; chunk < PARTITIONS_PER_GROUP; chunk += BLOCKSIZE)
    {
        // ---------------------------------------------------------
        // PHASE 1: Coalesced Read from Global -> Shared Memory
        // ---------------------------------------------------------
#pragma unroll
        for(int i = 0; i < BLOCKDIM; ++i)
        {
            int linear_idx    = i * BLOCKSIZE + tid;
            int partition_idx = linear_idx / BLOCKDIM;
            int element_idx   = linear_idx % BLOCKDIM;

            // Offset the global read by the current chunk
            int global_partition = group_start_partition + chunk + partition_idx;
            int global_in_idx    = global_partition * BLOCKDIM + element_idx;

            T val = pad_value;

            if(global_partition < nblocks && global_in_idx < m)
            {
                val = d_in[global_in_idx];
            }

            smem[partition_idx][element_idx] = val;
        }

        __syncthreads();

        // ---------------------------------------------------------
        // PHASE 2: Coalesced Write from Shared -> Global Memory
        // ---------------------------------------------------------
#pragma unroll
        for(int i = 0; i < BLOCKDIM; ++i)
        {
            int out_partition_idx = tid;
            int out_element_idx   = i;

            // CRITICAL CHANGE: Multiply by PARTITIONS_PER_GROUP to create the correct
            // 256-stride layout for the downstream kernel.
            // Add 'chunk' so the second pass writes to the correct offset.
            int linear_out = out_element_idx * PARTITIONS_PER_GROUP + chunk + out_partition_idx;

            if(group_start_partition + chunk + out_partition_idx < nblocks)
            {
                d_out[global_offset_out + linear_out] = smem[out_partition_idx][out_element_idx];
            }
        }

        // Synchronize before overwriting shared memory with the next chunk
        __syncthreads();
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
__device__ void
    data_untranspose_device_gemini(int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out)
{
    // Each CUDA block handles PARTITIONS_PER_GROUP partitions,
    // processed in smaller chunks dictated by BLOCKSIZE.
    const int group_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;
    const int nblocks = m_pad / BLOCKDIM;

    // Prevent completely out-of-bounds blocks from executing
    if(group_start_partition >= nblocks)
        return;

    __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

    const int tid = threadIdx.x;
    const int global_offset_pad = group_start_partition * BLOCKDIM;

    // Loop over the group in chunks of BLOCKSIZE
    for (int chunk = 0; chunk < PARTITIONS_PER_GROUP; chunk += BLOCKSIZE)
    {
        // ---------------------------------------------------------
        // PHASE 1: Coalesced Read from Padded Layout -> Shared Memory
        // ---------------------------------------------------------
#pragma unroll
        for(int i = 0; i < BLOCKDIM; ++i)
        {
            int in_partition_idx = tid;
            int in_element_idx   = i;

            // CRITICAL: Read using the PARTITIONS_PER_GROUP stride
            // generated by the forward marshaling kernel.
            int linear_in = in_element_idx * PARTITIONS_PER_GROUP + chunk + in_partition_idx;

            T val = T(0);
            if(group_start_partition + chunk + in_partition_idx < nblocks)
            {
                val = d_in[global_offset_pad + linear_in];
            }

            smem[in_partition_idx][in_element_idx] = val;
        }

        __syncthreads();

        // ---------------------------------------------------------
        // PHASE 2: Coalesced Write from Shared -> Global Standard Memory
        // ---------------------------------------------------------
#pragma unroll
        for(int i = 0; i < BLOCKDIM; ++i)
        {
            int linear_out = i * BLOCKSIZE + tid;
            int partition_idx = linear_out / BLOCKDIM;
            int element_idx   = linear_out % BLOCKDIM;

            int global_partition = group_start_partition + chunk + partition_idx;
            int global_out_idx   = global_partition * BLOCKDIM + element_idx;

            // Only write back to the unpadded, original system size 'm'.
            // The padded tail elements are naturally discarded.
            if(global_partition < nblocks && global_out_idx < m)
            {
                d_out[global_out_idx] = smem[partition_idx][element_idx];
            }
        }

        // Synchronize before overwriting shared memory with the next chunk
        __syncthreads();
    }
}




// template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
// __device__ void
//     data_transpose_device_gemini(int m, int m_pad, const T* __restrict__ d_in, T* __restrict__ d_out, T pad_value)
// {
//     // Each CUDA block processes exactly BLOCKSIZE partitions.
//     const int block_start_partition = blockIdx.x * PARTITIONS_PER_GROUP;

//     const int nblocks = m_pad / BLOCKDIM;

//     // Prevent completely out-of-bounds blocks from executing
//     if(block_start_partition >= nblocks)
//         return;

//     // 2D shared memory array sized dynamically at compile time.
//     // The "+1" pads the columns to prevent bank conflicts.
//     __shared__ T smem[BLOCKSIZE][BLOCKDIM + 1];

//     const int tid = threadIdx.x;

//     // ---------------------------------------------------------
//     // PHASE 1: Coalesced Read from Global -> Shared Memory
//     // ---------------------------------------------------------
// #pragma unroll
//     for(int i = 0; i < BLOCKDIM; ++i)
//     {
//         int linear_idx    = i * BLOCKSIZE + tid;
//         int partition_idx = linear_idx / BLOCKDIM;
//         int element_idx   = linear_idx % BLOCKDIM;

//         int global_partition = block_start_partition + partition_idx;
//         int global_in_idx    = global_partition * BLOCKDIM + element_idx;

//         // Initialize with the user-defined padding value (0.0 or 1.0)
//         T val = pad_value;

//         // Only read from d_in if within the original system size 'm'
//         // and within the padded partition bounds
//         if(global_partition < nblocks && global_in_idx < m)
//         {
//             val = d_in[global_in_idx];
//         }

//         smem[partition_idx][element_idx] = val;
//     }

//     __syncthreads();

//     // ---------------------------------------------------------
//     // PHASE 2: Coalesced Write from Shared -> Global Memory
//     // ---------------------------------------------------------
//     int global_offset_out = block_start_partition * BLOCKDIM;

// #pragma unroll
//     for(int i = 0; i < BLOCKDIM; ++i)
//     {
//         int out_partition_idx = tid;
//         int out_element_idx   = i;

//         int linear_out = out_element_idx * BLOCKSIZE + out_partition_idx;

//         // Write the padded/transposed data to the new m_pad sized array
//         if(block_start_partition + out_partition_idx < nblocks)
//         {
//             d_out[global_offset_out + linear_out] = smem[out_partition_idx][out_element_idx];
//         }
//     }
// }

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
__global__ void data_marshaling_kernel_gemini(int m,
                                            int m_pad,
                                            const T* __restrict__ lower,
                                            const T* __restrict__ main,
                                            const T* __restrict__ upper,
                                            T* __restrict__ lower_pad,
                                            T* __restrict__ main_pad,
                                            T* __restrict__ upper_pad)
{
    data_transpose_device_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
        m, m_pad, lower, lower_pad, static_cast<T>(0));
    __syncthreads();
    data_transpose_device_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
        m, m_pad, main, main_pad, static_cast<T>(1));
    __syncthreads();
    data_transpose_device_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
        m, m_pad, upper, upper_pad, static_cast<T>(0));
    __syncthreads();
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
__global__ void data_marshaling_B_kernel_gemini(
    int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        data_transpose_device_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, B + m * batch, B_pad + m_pad * batch, static_cast<T>(0));
        __syncthreads();
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
__global__ void reverse_data_marshaling_B_kernel_gemini(
    int m, int m_pad, int n, const T* __restrict__ B_pad, T* __restrict__ B)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        data_untranspose_device_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>(
            m, m_pad, B_pad + m_pad * batch, B + m * batch);
        __syncthreads();
    }
}

































template <uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y, uint32_t BLOCKDIM, typename T>
__device__ void data_transpose_device(
    int m, int m_pad, const T* __restrict__ data, T* __restrict__ data_pad, T val)
{
    // Consider lower_pad, main_pad, and upper_pad as BLOCKDIM x nblocks
    // column ordered 2d matrices. We want to transpose them.

    // Shared memory allocated for a 2D tile
    __shared__ T tile[BLOCKSIZE_Y][BLOCKSIZE_X + 1];

    const int nblocks = (m_pad / BLOCKDIM);

    // Global coordinates for reading from the input matrix
    const int row = blockIdx.x * BLOCKSIZE_X + threadIdx.x; // 0 ... BLOCKDIM-1
    const int col = blockIdx.y * BLOCKSIZE_Y + threadIdx.y; // 0 ... nblock-1

    // Load data from global memory into shared memory tile
    if(row < BLOCKDIM && col < nblocks)
    {
        tile[threadIdx.y][threadIdx.x]
            = ((col * BLOCKDIM + row) < m) ? data[col * BLOCKDIM + row] : val;
    }

    // Synchronize to ensure the entire tile is loaded
    __syncthreads();

    // Recompute global coordinates for writing the transposed matrix
    // Transpose the block dimensions: blockIdx.x and blockIdx.y swap roles
    const int out_row = blockIdx.y * BLOCKSIZE_Y + threadIdx.x;
    const int out_col = blockIdx.x * BLOCKSIZE_X + threadIdx.y;

    // Write the transposed data back to global memory coalesced
    if(out_row < nblocks && out_col < BLOCKDIM)
    {
        if((out_col * nblocks + out_row) < m_pad)
        {
            data_pad[out_col * nblocks + out_row] = tile[threadIdx.x][threadIdx.y];
        }
    }
}

template <uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y, uint32_t BLOCKDIM, typename T>
__global__ void data_marshaling_kernel_fast(int m,
                                            int m_pad,
                                            const T* __restrict__ lower,
                                            const T* __restrict__ main,
                                            const T* __restrict__ upper,
                                            T* __restrict__ lower_pad,
                                            T* __restrict__ main_pad,
                                            T* __restrict__ upper_pad)
{
    data_transpose_device<BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKDIM>(
        m, m_pad, lower, lower_pad, static_cast<T>(0));
    __syncthreads();
    data_transpose_device<BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKDIM>(
        m, m_pad, main, main_pad, static_cast<T>(1));
    __syncthreads();
    data_transpose_device<BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKDIM>(
        m, m_pad, upper, upper_pad, static_cast<T>(0));
    __syncthreads();
}

template <uint32_t BLOCKSIZE_X, uint32_t BLOCKSIZE_Y, uint32_t BLOCKDIM, typename T>
__global__ void data_marshaling_B_kernel_fast(
    int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
{
    for(int batch = blockIdx.z; batch < n; batch += gridDim.z)
    {
        data_transpose_device<BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKDIM>(
            m, m_pad, B + m * batch, B_pad + m_pad * batch, static_cast<T>(0));
    }
}

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

    const int lid = gid % (m_pad / BLOCKDIM);
    const int wid = gid / (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    lower_pad[gid] = (BLOCKDIM * lid + wid < m) ? lower[BLOCKDIM * lid + wid] : static_cast<T>(0);
    main_pad[gid]  = (BLOCKDIM * lid + wid < m) ? main[BLOCKDIM * lid + wid] : static_cast<T>(1);
    upper_pad[gid] = (BLOCKDIM * lid + wid < m) ? upper[BLOCKDIM * lid + wid] : static_cast<T>(0);
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void data_marshaling_B_device(
    int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int lid = gid % (m_pad / BLOCKDIM);
    const int wid = gid / (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    B_pad[gid] = (BLOCKDIM * lid + wid < m) ? B[BLOCKDIM * lid + wid] : static_cast<T>(0);
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void data_marshaling_B_kernel(
    int m, int m_pad, int n, const T* __restrict__ B, T* __restrict__ B_pad)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        data_marshaling_B_device<BLOCKSIZE, BLOCKDIM>(
            m, m_pad, n, B + m * batch, B_pad + m_pad * batch);
    }
}

template <typename T>
__host__ __device__ bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
{
    const T kappa = static_cast<T>(0.5) * (sqrt(static_cast<T>(5.0)) - static_cast<T>(1.0));

    T sigma = static_cast<T>(0);
    sigma   = max(static_cast<T>(abs(ak_1)), static_cast<T>(abs(ak_2)));
    sigma   = max(static_cast<T>(abs(bk_1)), sigma);
    sigma   = max(static_cast<T>(abs(ck)), sigma);
    sigma   = max(static_cast<T>(abs(ck_1)), sigma);

    return abs(bk) * sigma >= kappa * abs(ak_1 * ck);
}

template <uint32_t BLOCKDIM>
struct PivotMask
{
    unsigned int bits[(BLOCKDIM + 31) / 32];

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

    PivotMask<BLOCKDIM> pivot_mask{};

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

            const T wk = w[nblocks * k + gid];
            const T vk = v[nblocks * k + gid];

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

            const T wk   = w[nblocks * k + gid];
            const T wk_1 = w[nblocks * (k + 1) + gid];
            const T vk   = v[nblocks * k + gid];
            const T vk_1 = v[nblocks * (k + 1) + gid];

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
__device__ void LBMT_solve_rhs_device(int m_pad,
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

    PivotMask<BLOCKDIM> pivot_mask{};

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
            const T rhsk = rhs[nblocks * k + gid] * inv_bk;

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
            const T rhsk   = rhs[nblocks * k + gid] * det;
            const T rhsk_1 = rhs[nblocks * (k + 1) + gid] * det;

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

            rhs[nblocks * k + gid] += -tmp * rhs[nblocks * (k + 1) + gid];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[nblocks * k + gid];
            const T tmp2 = mt[nblocks * (k - 1) + gid];

            rhs[nblocks * k + gid] += -tmp1 * rhs[nblocks * (k + 1) + gid];
            rhs[nblocks * (k - 1) + gid] += -tmp2 * rhs[nblocks * (k + 1) + gid];

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
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        LBMT_solve_rhs_device<BLOCKSIZE, BLOCKDIM>(
            m_pad, n, lower, main, upper, mt, rhs + batch * m_pad);
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

    PivotMask<BLOCKDIM> pivot_mask{};

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

            const T wk = w[nblocks * k + gid];
            const T vk = v[nblocks * k + gid];

            w[nblocks * k + gid]  = wk * inv_bk;
            v[nblocks * k + gid]  = vk * inv_bk;
            mt[nblocks * k + gid] = ck * inv_bk;

            pivot_mask.set_pivoting_to_1x1(k);

            if(k < (BLOCKDIM - 1))
            {
                w[nblocks * (k + 1) + gid] += -ak_1 * wk * inv_bk;
            }

            // L * B * x = y
            const T rhsk = rhs[nblocks * k + gid] * inv_bk;

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

            const T wk   = w[nblocks * k + gid];
            const T wk_1 = w[nblocks * (k + 1) + gid];
            const T vk   = v[nblocks * k + gid];
            const T vk_1 = v[nblocks * (k + 1) + gid];

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
            const T rhsk   = rhs[nblocks * k + gid] * det;
            const T rhsk_1 = rhs[nblocks * (k + 1) + gid] * det;

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
            rhs[nblocks * k + gid] += -tmp * rhs[nblocks * (k + 1) + gid];

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
            rhs[nblocks * k + gid] += -tmp1 * rhs[nblocks * (k + 1) + gid];
            rhs[nblocks * (k - 1) + gid] += -tmp2 * rhs[nblocks * (k + 1) + gid];

            k -= 2;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void fill_s_matrix_device(int m_pad,
                                     int n,
                                     const T* __restrict__ w,
                                     const T* __restrict__ v,
                                     const T* __restrict__ rhs,
                                     T* __restrict__ S_lower,
                                     T* __restrict__ S_main,
                                     T* __restrict__ S_upper,
                                     T* __restrict__ S_rhs,
                                     bool write_s_matrix)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int s_size = 2 * m_pad / BLOCKDIM;

    if(write_s_matrix && gid < s_size)
    {
        S_upper[gid] = (gid % 2 == 0) ? v[gid / 2] : static_cast<T>(1);
        S_lower[gid]
            = (gid % 2 == 0) ? static_cast<T>(1) : w[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
    }

    if(write_s_matrix && gid >= 1 && gid < s_size - 1)
    {
        S_main[gid]
            = (gid % 2 == 0) ? w[gid / 2] : v[gid / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
    }

    if(gid < s_size / 2)
    {
        S_rhs[2 * gid]     = rhs[gid];
        S_rhs[2 * gid + 1] = rhs[gid + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
    }

    if(write_s_matrix && gid == 0)
    {
        S_lower[0] = static_cast<T>(0);
        S_main[0]  = static_cast<T>(1);
    }
    if(write_s_matrix && gid == 1)
    {
        S_lower[1] = static_cast<T>(0);
    }
    if(write_s_matrix && gid == s_size - 2)
    {
        S_upper[s_size - 2] = static_cast<T>(0);
    }
    if(write_s_matrix && gid == s_size - 1)
    {
        S_upper[s_size - 1] = static_cast<T>(0);
        S_main[s_size - 1]  = static_cast<T>(1);
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
    const int s_size = 2 * m_pad / BLOCKDIM;

    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        fill_s_matrix_device<BLOCKSIZE, BLOCKDIM>(m_pad,
                                                  n,
                                                  w,
                                                  v,
                                                  rhs + m_pad * batch,
                                                  S_lower,
                                                  S_main,
                                                  S_upper,
                                                  S_rhs + s_size * batch,
                                                  batch == 0);
    }
}

template <uint32_t S_SIZE, typename T>
__global__ void S_solve_kernel(int m,
                               int n,
                               const T* __restrict__ S_lower,
                               const T* __restrict__ S_main,
                               const T* __restrict__ S_upper,
                               T* __restrict__ S_rhs)
{
    static_assert(S_SIZE >= 2);

    const int batch = blockIdx.x;

    T mt[S_SIZE];

    PivotMask<S_SIZE> pivot_mask{};

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
            const T rhsk = S_rhs[k + m * batch] * inv_bk;

            S_rhs[k + m * batch] = rhsk;

            if(k < (S_SIZE - 1))
            {
                S_rhs[k + 1 + m * batch] += -(ak_1 * rhsk);

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
            const T rhsk   = S_rhs[k + m * batch] * det;
            const T rhsk_1 = S_rhs[k + 1 + m * batch] * det;

            S_rhs[k + m * batch]     = (bk_1 * rhsk - ck * rhsk_1);
            S_rhs[k + 1 + m * batch] = (-ak_1 * rhsk + bk * rhsk_1);

            if(k < (S_SIZE - 2))
            {
                S_rhs[k + 2 + m * batch] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

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

            S_rhs[k + m * batch] += -tmp * S_rhs[k + 1 + m * batch];

            k -= 1;
        }
        else
        {
            const T tmp1 = mt[k];
            const T tmp2 = mt[k - 1];

            S_rhs[k + m * batch] += -tmp1 * S_rhs[k + 1 + m * batch];
            S_rhs[k - 1 + m * batch] += -tmp2 * S_rhs[k + 1 + m * batch];

            k -= 2;
        }
    }
}

// template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
// __device__ void backward_solve_device(
//     int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ rhs)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int gid = tid + BLOCKSIZE * bid;

//     const int nblocks = m_pad / BLOCKDIM;

//     if(gid >= nblocks)
//     {
//         return;
//     }

//     // backward solve (S * x = B_pad)
//     const T x1 = (gid >= 1) ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (gid - 1)] : static_cast<T>(0);
//     const T x2 = (gid < (m_pad / BLOCKDIM - 1)) ? rhs[gid + 1] : static_cast<T>(0);

//     for(int j = 1; j < BLOCKDIM - 1; j++)
//     {
//         rhs[(m_pad / BLOCKDIM) * j + gid] = rhs[(m_pad / BLOCKDIM) * j + gid]
//                                             - w[(m_pad / BLOCKDIM) * j + gid] * x1
//                                             - v[(m_pad / BLOCKDIM) * j + gid] * x2;
//     }
// }
template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void backward_solve_device(
    int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int lid = gid % (m_pad / BLOCKDIM);
    const int wid = gid / (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    // backward solve (S * x = B_pad)
    const T x1
        = (lid >= 1) ? B_pad[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (lid - 1)] : static_cast<T>(0);
    const T x2 = (lid < (m_pad / BLOCKDIM - 1)) ? B_pad[lid + 1] : static_cast<T>(0);

    if(wid >= 1 && wid < BLOCKDIM - 1)
    {
        B_pad[(m_pad / BLOCKDIM) * wid + lid] = B_pad[(m_pad / BLOCKDIM) * wid + lid]
                                                - w[(m_pad / BLOCKDIM) * wid + lid] * x1
                                                - v[(m_pad / BLOCKDIM) * wid + lid] * x2;
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void backward_solve_kernel(
    int m_pad, int n, const T* __restrict__ w, const T* __restrict__ v, T* __restrict__ B_pad)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        backward_solve_device<BLOCKSIZE, BLOCKDIM>(m_pad, n, w, v, B_pad + m_pad * batch);
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void data_marshaling_device3(int m,
                                        int m_pad,
                                        int n,
                                        const T* __restrict__ w,
                                        const T* __restrict__ v,
                                        const T* __restrict__ B_pad,
                                        T* __restrict__ B)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int lid = gid % (m_pad / BLOCKDIM);
    const int wid = gid / (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    if(BLOCKDIM * lid + wid < m)
    {
        // backward solve (S * x = B_pad)
        const T x1 = (lid >= 1) ? B_pad[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (lid - 1)]
                                : static_cast<T>(0);
        const T x2 = (lid < (m_pad / BLOCKDIM - 1)) ? B_pad[lid + 1] : static_cast<T>(0);

        if(wid >= 1 && wid < BLOCKDIM - 1)
        {
            B[BLOCKDIM * lid + wid] = B_pad[(m_pad / BLOCKDIM) * wid + lid]
                                      - w[(m_pad / BLOCKDIM) * wid + lid] * x1
                                      - v[(m_pad / BLOCKDIM) * wid + lid] * x2;
        }
        else
        {
            B[BLOCKDIM * lid + wid] = B_pad[gid];
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void data_marshaling_kernel3(int m,
                                        int m_pad,
                                        int n,
                                        const T* __restrict__ w,
                                        const T* __restrict__ v,
                                        const T* __restrict__ B_pad,
                                        T* __restrict__ B)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        data_marshaling_device3<BLOCKSIZE, BLOCKDIM>(
            m, m_pad, n, w, v, B_pad + m_pad * batch, B + m * batch);
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void
    data_marshaling_device2(int m, int m_pad, int n, const T* __restrict__ B_pad, T* __restrict__ B)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + BLOCKSIZE * bid;

    const int lid = gid % (m_pad / BLOCKDIM);
    const int wid = gid / (m_pad / BLOCKDIM);

    if(gid >= m_pad)
    {
        return;
    }

    if(BLOCKDIM * lid + wid < m)
    {
        B[BLOCKDIM * lid + wid] = B_pad[gid];
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void
    data_marshaling_kernel2(int m, int m_pad, int n, const T* __restrict__ B_pad, T* __restrict__ B)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        data_marshaling_device2<BLOCKSIZE, BLOCKDIM>(
            m, m_pad, n, B_pad + m_pad * batch, B + m * batch);
    }
}

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
//     T temp      = h_y[i];
//     h_y[i]      = h_y[i + 1];
//     h_y[i + 1]  = temp;
// }
// for(int i = 0; i < s_size / 2; i++)
// {
//     h_B_pad[i]                                       = h_y[2 * i];
//     h_B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)] = h_y[2 * i + 1];
// }
template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__device__ void scatter_S_B_to_B_pad_device(
    int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
{
    const int i = blockIdx.x * BLOCKSIZE + threadIdx.x; // [0, s_size/2)

    if(i >= s_size / 2)
        return;

    const int stride = (m_pad / BLOCKDIM) * (BLOCKDIM - 1);

    // After the swap loop, element at position 2*i is:
    //   i == 0  -> S_B[0]       (not touched by the swap)
    //   i  > 0  -> S_B[2*i - 1] (position 2*i was swapped with 2*i-1)
    const T val_even = (i == 0) ? S_B[0] : S_B[2 * i - 1];

    // After the swap loop, element at position 2*i+1 is:
    //   2*i+1 < s_size-1  -> S_B[2*i + 2] (swapped with its right neighbour)
    //   2*i+1 == s_size-1 -> S_B[s_size-1] (last element, not touched)
    const T val_odd = (2 * i + 1 < s_size - 1) ? S_B[2 * i + 2] : S_B[s_size - 1];

    B_pad[i]          = val_even;
    B_pad[i + stride] = val_odd;
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void scatter_S_B_to_B_pad_kernel(
    int s_size, int m_pad, int n, const T* __restrict__ S_B, T* __restrict__ B_pad)
{
    for(int batch = blockIdx.y; batch < n; batch += gridDim.y)
    {
        scatter_S_B_to_B_pad_device<BLOCKSIZE, BLOCKDIM>(
            s_size, m_pad, n, S_B + batch * s_size, B_pad + batch * m_pad);
    }
}

template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
__global__ void S_solve_fused_kernel(const T* __restrict__ lower,
                                     const T* __restrict__ main,
                                     const T* __restrict__ upper,
                                     T* __restrict__ rhs)
{
    static_assert(BLOCKDIM >= 2);
    static_assert(BLOCKDIM % 2 == 0);
    static_assert(BLOCKSIZE % 2 == 0);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch = bid;

    constexpr int M = BLOCKSIZE * BLOCKDIM;

    // Transpose data
    __shared__ T slower[M];
    __shared__ T smain[M];
    __shared__ T supper[M];
    __shared__ T srhs[M];

    for(int i = 0; i < BLOCKDIM; i++)
    {
        slower[BLOCKSIZE * i + tid] = lower[BLOCKDIM * tid + i];
        smain[BLOCKSIZE * i + tid]  = main[BLOCKDIM * tid + i];
        supper[BLOCKSIZE * i + tid] = upper[BLOCKDIM * tid + i];
        srhs[BLOCKSIZE * i + tid]   = rhs[BLOCKDIM * tid + i + M * batch];
    }

    __syncthreads();

    // LBMT solve
    __shared__ T sw[M];
    __shared__ T sv[M];
    __shared__ T smt[M];

    T bk = smain[tid];

    PivotMask<BLOCKDIM> pivot_mask{};

    sw[tid]                              = slower[tid];
    sv[tid + (BLOCKDIM - 1) * BLOCKSIZE] = supper[tid + (BLOCKDIM - 1) * BLOCKSIZE];

    int k = 0;
    while(k < BLOCKDIM)
    {
        T ck   = supper[BLOCKSIZE * k + tid];
        T ck_1 = (k < (BLOCKDIM - 1)) ? supper[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
        T bk_1 = (k < (BLOCKDIM - 1)) ? smain[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
        T ak_1 = (k < (BLOCKDIM - 1)) ? slower[BLOCKSIZE * (k + 1) + tid] : static_cast<T>(0);
        T ak_2 = (k < (BLOCKDIM - 2)) ? slower[BLOCKSIZE * (k + 2) + tid] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

        // 1x1 pivoting
        if(use_1x1_pivot || k == (BLOCKDIM - 1))
        {
            const T inv_bk = static_cast<T>(1) / bk;

            const T wk = sw[BLOCKSIZE * k + tid];
            const T vk = sv[BLOCKSIZE * k + tid];

            sw[BLOCKSIZE * k + tid]  = wk * inv_bk;
            sv[BLOCKSIZE * k + tid]  = vk * inv_bk;
            smt[BLOCKSIZE * k + tid] = ck * inv_bk;

            pivot_mask.set_pivoting_to_1x1(k);

            if(k < (BLOCKDIM - 1))
            {
                sw[BLOCKSIZE * (k + 1) + tid] += -ak_1 * wk * inv_bk;
            }

            // L * B * x = y
            const T rhsk = srhs[BLOCKSIZE * k + tid] * inv_bk;

            srhs[BLOCKSIZE * k + tid] = rhsk;

            if(k < (BLOCKDIM - 1))
            {
                srhs[BLOCKSIZE * (k + 1) + tid] += -(ak_1 * rhsk);

                bk_1 = bk_1 - ak_1 * ck * inv_bk;
            }

            bk = bk_1;

            k += 1;
        }
        else
        {
            const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

            const T wk   = sw[BLOCKSIZE * k + tid];
            const T wk_1 = sw[BLOCKSIZE * (k + 1) + tid];
            const T vk   = sv[BLOCKSIZE * k + tid];
            const T vk_1 = sv[BLOCKSIZE * (k + 1) + tid];

            sw[BLOCKSIZE * k + tid]  = (bk_1 * wk - ck * wk_1) * det;
            sv[BLOCKSIZE * k + tid]  = (bk_1 * vk - ck * vk_1) * det;
            smt[BLOCKSIZE * k + tid] = -ck * ck_1 * det;

            pivot_mask.set_pivoting_to2x2(k);

            if(k < (BLOCKDIM - 1))
            {
                sw[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * wk + bk * wk_1) * det;
                sv[BLOCKSIZE * (k + 1) + tid]  = (-ak_1 * vk + bk * vk_1) * det;
                smt[BLOCKSIZE * (k + 1) + tid] = bk * ck_1 * det;

                pivot_mask.set_pivoting_to2x2(k + 1);
            }

            T bk_2 = static_cast<T>(0);

            if(k < (BLOCKDIM - 2))
            {
                sw[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
            }

            // |bk   ck  ||xk  |   |rhsk   |
            // |ak_1 bk_1||xk_1| = |rhsk _1|
            //
            //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
            //                                  |-ak_1  bk |

            // L * B * x = y
            const T rhsk   = srhs[BLOCKSIZE * k + tid] * det;
            const T rhsk_1 = srhs[BLOCKSIZE * (k + 1) + tid] * det;

            srhs[BLOCKSIZE * k + tid]       = (bk_1 * rhsk - ck * rhsk_1);
            srhs[BLOCKSIZE * (k + 1) + tid] = (-ak_1 * rhsk + bk * rhsk_1);

            if(k < (BLOCKDIM - 2))
            {
                srhs[BLOCKSIZE * (k + 2) + tid] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                bk_2 = smain[BLOCKSIZE * (k + 2) + tid];
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
            const T tmp = smt[BLOCKSIZE * k + tid];

            sw[BLOCKSIZE * k + tid] += -tmp * sw[BLOCKSIZE * (k + 1) + tid];
            sv[BLOCKSIZE * k + tid] += -tmp * sv[BLOCKSIZE * (k + 1) + tid];
            srhs[BLOCKSIZE * k + tid] += -tmp * srhs[BLOCKSIZE * (k + 1) + tid];

            k -= 1;
        }
        else
        {
            const T tmp1 = smt[BLOCKSIZE * k + tid];
            const T tmp2 = smt[BLOCKSIZE * (k - 1) + tid];

            sw[BLOCKSIZE * k + tid] += -tmp1 * sw[BLOCKSIZE * (k + 1) + tid];
            sw[BLOCKSIZE * (k - 1) + tid] += -tmp2 * sw[BLOCKSIZE * (k + 1) + tid];
            sv[BLOCKSIZE * k + tid] += -tmp1 * sv[BLOCKSIZE * (k + 1) + tid];
            sv[BLOCKSIZE * (k - 1) + tid] += -tmp2 * sv[BLOCKSIZE * (k + 1) + tid];
            srhs[BLOCKSIZE * k + tid] += -tmp1 * srhs[BLOCKSIZE * (k + 1) + tid];
            srhs[BLOCKSIZE * (k - 1) + tid] += -tmp2 * srhs[BLOCKSIZE * (k + 1) + tid];

            k -= 2;
        }
    }

    __syncthreads();

    // Fill S system
    constexpr int S_SIZE = 2 * BLOCKSIZE;

    __shared__ T S_lower[S_SIZE];
    __shared__ T S_main[S_SIZE];
    __shared__ T S_upper[S_SIZE];
    __shared__ T S_rhs[S_SIZE];

    for(int i = 0; i < 2; i++)
    {
        const int gid = BLOCKSIZE * i + tid;

        S_upper[gid] = (gid % 2 == 0) ? sv[gid / 2] : static_cast<T>(1);
        S_lower[gid]
            = (gid % 2 == 0) ? static_cast<T>(1) : sw[gid / 2 + BLOCKSIZE * (BLOCKDIM - 1)];

        if(gid >= 1 && gid < S_SIZE - 1)
        {
            S_main[gid] = (gid % 2 == 0) ? sw[gid / 2] : sv[gid / 2 + BLOCKSIZE * (BLOCKDIM - 1)];
        }

        if(gid < S_SIZE / 2)
        {
            S_rhs[2 * gid]     = srhs[gid];
            S_rhs[2 * gid + 1] = srhs[gid + BLOCKSIZE * (BLOCKDIM - 1)];
        }
    }

    if(tid == 0)
    {
        S_lower[0] = static_cast<T>(0);
        S_main[0]  = static_cast<T>(1);
    }
    if(tid == 1)
    {
        S_lower[1] = static_cast<T>(0);
    }
    if(tid == S_SIZE - BLOCKSIZE - 2)
    {
        S_upper[S_SIZE - 2] = static_cast<T>(0);
    }
    if(tid == S_SIZE - BLOCKSIZE - 1)
    {
        S_upper[S_SIZE - 1] = static_cast<T>(0);
        S_main[S_SIZE - 1]  = static_cast<T>(1);
    }

    __syncthreads();

    // Solve S system (solved by 1 thread)
    if(tid == 0)
    {
        __shared__ T S_mt[S_SIZE];

        PivotMask<S_SIZE> pivot_mask{};

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

                S_mt[k] = ck * inv_bk;

                pivot_mask.set_pivoting_to_1x1(k);

                // L * B * x = y
                const T rhsk = S_rhs[k] * inv_bk;

                S_rhs[k] = rhsk;

                if(k < (S_SIZE - 1))
                {
                    S_rhs[k + 1] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                S_mt[k] = -ck * ck_1 * det;

                pivot_mask.set_pivoting_to2x2(k);

                if(k < (S_SIZE - 1))
                {
                    S_mt[k + 1] = bk * ck_1 * det;

                    pivot_mask.set_pivoting_to2x2(k + 1);
                }

                T bk_2 = static_cast<T>(0);

                // L * B * x = y
                const T rhsk   = S_rhs[k] * det;
                const T rhsk_1 = S_rhs[k + 1] * det;

                S_rhs[k]     = (bk_1 * rhsk - ck * rhsk_1);
                S_rhs[k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (S_SIZE - 2))
                {
                    S_rhs[k + 2] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

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
                const T tmp = S_mt[k];

                S_rhs[k] += -tmp * S_rhs[k + 1];

                k -= 1;
            }
            else
            {
                const T tmp1 = S_mt[k];
                const T tmp2 = S_mt[k - 1];

                S_rhs[k] += -tmp1 * S_rhs[k + 1];
                S_rhs[k - 1] += -tmp2 * S_rhs[k + 1];

                k -= 2;
            }
        }
    }

    __syncthreads();

    // Scatter
    constexpr int STRIDE = BLOCKSIZE * (BLOCKDIM - 1);

    // After the swap loop, element at position 2*tid is:
    //   tid == 0  -> S_rhs[0]       (not touched by the swap)
    //   tid  > 0  -> S_rhs[2*tid - 1] (position 2*tid was swapped with 2*tid-1)
    const T val_even = (tid == 0) ? S_rhs[0] : S_rhs[2 * tid - 1];

    // After the swap loop, element at position 2*i+1 is:
    //   2*tid+1 < S_SIZE-1  -> S_rhs[2*tid + 2] (swapped with its right neighbour)
    //   2*tid+1 == S_SIZE-1 -> S_rhs[S_SIZE-1] (last element, not touched)
    const T val_odd = (2 * tid + 1 < S_SIZE - 1) ? S_rhs[2 * tid + 2] : S_rhs[S_SIZE - 1];

    srhs[tid]          = val_even;
    srhs[tid + STRIDE] = val_odd;

    __syncthreads();

    // backward solve (S * x = rhs)
    const T x1 = (tid >= 1) ? srhs[BLOCKSIZE * (BLOCKDIM - 1) + (tid - 1)] : static_cast<T>(0);
    const T x2 = (tid < (BLOCKSIZE - 1)) ? srhs[tid + 1] : static_cast<T>(0);

    for(int j = 1; j < BLOCKDIM - 1; j++)
    {
        srhs[BLOCKSIZE * j + tid] = srhs[BLOCKSIZE * j + tid] - sw[BLOCKSIZE * j + tid] * x1
                                    - sv[BLOCKSIZE * j + tid] * x2;
    }

    __syncthreads();

    // Transpose back
    for(int i = 0; i < BLOCKDIM; i++)
    {
        rhs[BLOCKDIM * tid + i + M * batch] = srhs[BLOCKSIZE * i + tid];
    }
}

#endif // TRIDIAGONAL_SOLVER_SPIKE_KERNELS_H
