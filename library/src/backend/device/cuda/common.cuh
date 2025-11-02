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

#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

namespace linalg
{
    template <typename T>
    __host__ __device__ __forceinline__ T abs(T val)
    {
        return val < 0 ? -val : val;
    }

    template <typename T>
    __host__ __device__ __forceinline__ T max(T val1, T val2)
    {
        return val1 < val2 ? val2 : val1;
    }

    template <typename T>
    __host__ __device__ __forceinline__ T min(T val1, T val2)
    {
        return val1 < val2 ? val1 : val2;
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

template <uint32_t WARPSIZE, typename T>
__device__ __forceinline__ void warp_reduction_sum(T* __restrict__ data, int lid)
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
__device__ __forceinline__ void block_reduction_sum(T* __restrict__ data, int tid)
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
__device__ __forceinline__ void block_reduction_max(T* __restrict__ data, int tid)
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
__device__ __forceinline__ void block_reduction_min(T* __restrict__ data, int tid)
{
    if(BLOCKSIZE > 512)
    {
        if(tid < 512)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[512 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 256)
    {
        if(tid < 256)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[256 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 128)
    {
        if(tid < 128)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[128 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 64)
    {
        if(tid < 64)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[64 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 32)
    {
        if(tid < 32)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[32 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 16)
    {
        if(tid < 16)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[16 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 8)
    {
        if(tid < 8)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[8 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 4)
    {
        if(tid < 4)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[4 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 2)
    {
        if(tid < 2)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[2 + tid]));
        }
        __syncthreads();
    }
    if(BLOCKSIZE > 1)
    {
        if(tid < 1)
        {
            data[tid] = linalg::min(linalg::abs(data[tid]), linalg::abs(data[1 + tid]));
        }
        __syncthreads();
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void fill_kernel(T* __restrict__ data, size_t size, T val)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        data[tid] = val;
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void copy_kernel(T* __restrict__ dest, const T* __restrict__ src, size_t size)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        dest[tid] = src[tid];
    }
}

template <uint32_t HASH_TABLE_SIZE>
__device__ bool atomic_insert_key(int* key_table, int key, int EMPTY_VAL)
{
    // 1. Calculate the initial hash index
    uint32_t start_index = key % HASH_TABLE_SIZE;
    uint32_t index       = start_index;

    // 2. Linear Probing with an atomic insertion
    for(uint32_t i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        // Calculate the probe index (circular)
        index = (start_index + i) % HASH_TABLE_SIZE;

        // Try to atomically replace the EMPTY_VAL with the new key.
        // atomicCAS(address, compare, val)
        // Checks if *address == compare. If true, sets *address = val and returns compare.
        // If false, returns *address without changing it.
        int old_val = atomicCAS(&key_table[index], EMPTY_VAL, key);

        if(old_val == EMPTY_VAL)
        {
            // Success: The slot was empty and we inserted the key.
            return true;
        }
        else if(old_val == key)
        {
            // Key already exists (handling duplicates in linear probing)
            // Return false to indicate that no insertion was required
            return false;
        }
    }

    // Failure: Table is full after probing all slots
    return false;
}

template <uint32_t HASH_TABLE_SIZE, typename T>
__device__ bool
    atomic_insert_key_value(int* key_table, T* values_table, int key, T value, int EMPTY_VAL)
{
    // 1. Calculate the initial hash index
    uint32_t start_index = key % HASH_TABLE_SIZE;
    uint32_t index       = start_index;

    // 2. Linear Probing with an atomic insertion
    for(uint32_t i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        // Calculate the probe index (circular)
        index = (start_index + i) % HASH_TABLE_SIZE;

        // Try to atomically replace the EMPTY_VAL with the new key.
        // atomicCAS(address, compare, val)
        // Checks if *address == compare. If true, sets *address = val and returns compare.
        // If false, returns *address without changing it.
        int old_val = atomicCAS(&key_table[index], EMPTY_VAL, key);

        __threadfence_block();

        if(old_val == EMPTY_VAL)
        {
            // Success: The slot was empty and we inserted the key.
            // values_table[index] = value;
            atomicAdd(&values_table[index], value);
            return true;
        }
        else if(old_val == key)
        {
            // Key already exists (handling duplicates in linear probing)
            // Return false to indicate that no insertion was required
            atomicAdd(&values_table[index], value);
            return false;
        }
        __threadfence_block();
    }

    // Failure: Table is full after probing all slots
    return false;
}

#endif