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
#include <assert.h>
#include <stdint.h>

#include "host_memory.h"

#include "../../trace.h"

namespace linalg
{
    //-------------------------------------------------------------------------------
    // fill array
    //-------------------------------------------------------------------------------
    template <typename T>
    static void host_fill_impl(T* x, size_t n, T value)
    {
        ROUTINE_TRACE("host_fill_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < n; i++)
        {
            x[i] = value;
        }
    }

    //-------------------------------------------------------------------------------
    // copy array
    //-------------------------------------------------------------------------------
    template <typename T>
    static void host_copy_impl(T* dest, const T* src, size_t n)
    {
        ROUTINE_TRACE("host_copy_impl");

        assert(dest != src);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < n; i++)
        {
            dest[i] = src[i];
        }
    }
}

template <typename T>
void linalg::copy_h2h(T* dest, const T* src, size_t size)
{
    host_copy_impl(dest, src, size);
}

template <typename T>
void linalg::host_fill(T* data, size_t size, T val)
{
    host_fill_impl(data, size, val);
}

template void linalg::copy_h2h<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::copy_h2h<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::copy_h2h<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::copy_h2h<double>(double* dest, const double* src, size_t size);

template void linalg::host_fill<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void linalg::host_fill<int32_t>(int32_t* data, size_t size, int32_t val);
template void linalg::host_fill<int64_t>(int64_t* data, size_t size, int64_t val);
template void linalg::host_fill<double>(double* data, size_t size, double val);