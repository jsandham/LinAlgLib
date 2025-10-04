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
#include "device_primitives.h"
#include <cmath>
#include <cstdint>
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_primitives.h"
#endif

// Find minimum
template<typename T>
T linalg::device_find_min(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::device_find_min");

    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_find_minimum(x.get_size(), x.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return static_cast<T>(0);
    }
}

// Find maximum
template<typename T>
T linalg::device_find_max(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::device_find_max");

    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_find_maximum(x.get_size(), x.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return static_cast<T>(0);
    }
}

// Exclusive scan
template <typename T>
void linalg::device_exclusive_scan(vector<T>& x)
{
    ROUTINE_TRACE("linalg::device_exclusive_scan");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_exclusive_scan((int)x.get_size(), x.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

template int32_t linalg::device_find_min<int32_t>(const vector<int32_t>& x);
template int64_t linalg::device_find_min<int64_t>(const vector<int64_t>& x);
template double linalg::device_find_min<double>(const vector<double>& x);

template int32_t linalg::device_find_max<int32_t>(const vector<int32_t>& x);
template int64_t linalg::device_find_max<int64_t>(const vector<int64_t>& x);
template double linalg::device_find_max<double>(const vector<double>& x);

template void linalg::device_exclusive_scan<int32_t>(vector<int32_t>& x);
template void linalg::device_exclusive_scan<int64_t>(vector<int64_t>& x);
template void linalg::device_exclusive_scan<double>(vector<double>& x);