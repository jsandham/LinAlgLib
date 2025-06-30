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

#include "cuda_kernels.h"

template <uint32_t BLOCKSIZE, typename T>
static __global__ void fill_kernel(T* data, size_t size, T val)
{
    int tid = threadIdx.x + BLOCKSIZE * blockIdx.x;

    if(tid < size)
    {
        data[tid] = val;
    }
}


template <typename T> 
void launch_cuda_fill_kernel(T* data, size_t size, T val)
{
    fill_kernel<256><<<((size - 1) / 256 + 1), 256>>>(data, size, val);
}

template void launch_cuda_fill_kernel<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void launch_cuda_fill_kernel<int32_t>(int32_t* data, size_t size, int32_t val);
template void launch_cuda_fill_kernel<int64_t>(int64_t* data, size_t size, int64_t val);
template void launch_cuda_fill_kernel<double>(double* data, size_t size, double val);
