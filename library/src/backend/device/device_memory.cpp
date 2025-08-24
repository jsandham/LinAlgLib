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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
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
#include "device_memory.h"
#include <cstdint>
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_memory.h"
#endif

template <typename T>
void linalg::device_allocate(T** ptr, size_t size)
{
    ROUTINE_TRACE("linalg::device_allocate");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_allocate(ptr, size));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template <typename T>
void linalg::device_free(T* ptr)
{
    ROUTINE_TRACE("linalg::device_free");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_free(ptr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template <typename T>
void linalg::copy_h2d(T* dest, const T* src, size_t size)
{
    ROUTINE_TRACE("linalg::copy_h2d");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_copy_h2d(dest, src, size));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template <typename T>
void linalg::copy_d2h(T* dest, const T* src, size_t size)
{
    ROUTINE_TRACE("linalg::copy_d2h");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_copy_d2h(dest, src, size));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template <typename T>
void linalg::copy_d2d(T* dest, const T* src, size_t size)
{
    ROUTINE_TRACE("linalg::copy_d2d");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_copy_d2d(dest, src, size));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template <typename T>
void linalg::device_fill(T* data, size_t size, T val)
{
    ROUTINE_TRACE("linalg::device_fill");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_fill(data, size, val));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template void linalg::device_allocate<uint32_t>(uint32_t** ptr, size_t size);
template void linalg::device_allocate<int32_t>(int32_t** ptr, size_t size);
template void linalg::device_allocate<int64_t>(int64_t** ptr, size_t size);
template void linalg::device_allocate<double>(double** ptr, size_t size);

template void linalg::device_free<uint32_t>(uint32_t* ptr);
template void linalg::device_free<int32_t>(int32_t* ptr);
template void linalg::device_free<int64_t>(int64_t* ptr);
template void linalg::device_free<double>(double* ptr);

template void linalg::copy_h2d<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::copy_h2d<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::copy_h2d<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::copy_h2d<double>(double* dest, const double* src, size_t size);

template void linalg::copy_d2h<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::copy_d2h<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::copy_d2h<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::copy_d2h<double>(double* dest, const double* src, size_t size);

template void linalg::copy_d2d<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::copy_d2d<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::copy_d2d<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::copy_d2d<double>(double* dest, const double* src, size_t size);

template void linalg::device_fill<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void linalg::device_fill<int32_t>(int32_t* data, size_t size, int32_t val);
template void linalg::device_fill<int64_t>(int64_t* data, size_t size, int64_t val);
template void linalg::device_fill<double>(double* data, size_t size, double val);