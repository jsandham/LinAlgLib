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

#ifndef TRIDIAGONAL_THOMAS_ALGORITHM_KERNELS_H
#define TRIDIAGONAL_THOMAS_ALGORITHM_KERNELS_H

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

#endif // TRIDIAGONAL_THOMAS_ALGORITHM_KERNELS_H
