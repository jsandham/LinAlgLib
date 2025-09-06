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
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>

#include "common.cuh"
#include "cuda_memory.h"

#include "../../../trace.h"

template <typename T>
void linalg::cuda_allocate(T** ptr, size_t size)
{
    CHECK_CUDA(cudaMalloc((void**)ptr, sizeof(T) * size));
}

template <typename T>
void linalg::cuda_free(T* ptr)
{
    CHECK_CUDA(cudaFree(ptr));
}

template <typename T>
void linalg::cuda_copy_h2d(T* dest, const T* src, size_t size)
{
    CHECK_CUDA(cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template <typename T>
void linalg::cuda_copy_d2h(T* dest, const T* src, size_t size)
{
    CHECK_CUDA(cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
void linalg::cuda_copy_d2d(T* dest, const T* src, size_t size)
{
    CHECK_CUDA(cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template <typename T>
void linalg::cuda_fill(T* data, size_t size, T val)
{
    if(val == static_cast<T>(0))
    {
        CHECK_CUDA(cudaMemset(data, 0, sizeof(T) * size));
    }
    else
    {
        ROUTINE_TRACE("launch_cuda_fill_kernel");
        fill_kernel<256><<<((size - 1) / 256 + 1), 256>>>(data, size, val);
        CHECK_CUDA_LAUNCH_ERROR();
    }
}

template void linalg::cuda_allocate<uint32_t>(uint32_t** ptr, size_t size);
template void linalg::cuda_allocate<int32_t>(int32_t** ptr, size_t size);
template void linalg::cuda_allocate<int64_t>(int64_t** ptr, size_t size);
template void linalg::cuda_allocate<double>(double** ptr, size_t size);

template void linalg::cuda_free<uint32_t>(uint32_t* ptr);
template void linalg::cuda_free<int32_t>(int32_t* ptr);
template void linalg::cuda_free<int64_t>(int64_t* ptr);
template void linalg::cuda_free<double>(double* ptr);

template void linalg::cuda_copy_h2d<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::cuda_copy_h2d<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::cuda_copy_h2d<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::cuda_copy_h2d<double>(double* dest, const double* src, size_t size);

template void linalg::cuda_copy_d2h<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::cuda_copy_d2h<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::cuda_copy_d2h<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::cuda_copy_d2h<double>(double* dest, const double* src, size_t size);

template void linalg::cuda_copy_d2d<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::cuda_copy_d2d<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::cuda_copy_d2d<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::cuda_copy_d2d<double>(double* dest, const double* src, size_t size);

template void linalg::cuda_fill<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void linalg::cuda_fill<int32_t>(int32_t* data, size_t size, int32_t val);
template void linalg::cuda_fill<int64_t>(int64_t* data, size_t size, int64_t val);
template void linalg::cuda_fill<double>(double* data, size_t size, double val);