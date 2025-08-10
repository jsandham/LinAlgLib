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
#include <stdint.h>
#include <assert.h>

#include "host_memory.h"

#include "../../trace.h"

namespace linalg
{
    //-------------------------------------------------------------------------------
    // fill array
    //-------------------------------------------------------------------------------
    template <typename T>
    static void host_fill(T* __restrict x, size_t n, T value)
    {
        ROUTINE_TRACE("host_fill_with_zeros");

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
    static void host_copy(T* __restrict dest, const T* __restrict src, size_t n)
    {
        ROUTINE_TRACE("host_copy");

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
void linalg::host::copy_h2h(T* dest, const T* src, size_t size)
{
    host_copy(dest, src, size);
}

template <typename T>
void linalg::host::fill(T* data, size_t size, T val)
{
    host_fill(data, size, val);
}

template void linalg::host::copy_h2h<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void linalg::host::copy_h2h<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void linalg::host::copy_h2h<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void linalg::host::copy_h2h<double>(double* dest, const double* src, size_t size);

template void linalg::host::fill<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void linalg::host::fill<int32_t>(int32_t* data, size_t size, int32_t val);
template void linalg::host::fill<int64_t>(int64_t* data, size_t size, int64_t val);
template void linalg::host::fill<double>(double* data, size_t size, double val);