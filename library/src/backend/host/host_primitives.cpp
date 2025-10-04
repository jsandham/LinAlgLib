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

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../trace.h"
#include "host_primitives.h"

namespace linalg
{
    //-------------------------------------------------------------------------------
    // find_min
    //-------------------------------------------------------------------------------
    template <typename T>
    static T host_find_min_impl(const T* x, int n)
    {
        ROUTINE_TRACE("host_find_min_impl");

        T min = std::numeric_limits<T>::max();
        for(int i = 0; i < n; i++)
        {
            min = std::min(min, std::abs(x[i]));
        }

        return min;
    }

    //-------------------------------------------------------------------------------
    // find_max
    //-------------------------------------------------------------------------------
    template<typename T>
    static T host_find_max_impl(const T* x, int n)
    {
        ROUTINE_TRACE("host_find_max_impl");

        T max = std::numeric_limits<T>::min();
        for(int i = 0; i < n; i++)
        {
            max = std::max(max, std::abs(x[i]));
        }

        return max;
    }

    //-------------------------------------------------------------------------------
    // exclusive scan
    //-------------------------------------------------------------------------------
    template<typename T>
    static void host_exclusive_scan_impl(T* x, int n)
    {
        ROUTINE_TRACE("host_exclusive_scan_impl");

        if(n > 0)
        {
            T sum = 0;
            for(int i = 0; i < n; i++)
            {
                T temp = x[i];
                x[i]         = sum;
                sum += temp;
            }
        }
    }
}

// Find minimum
template<typename T>
T linalg::host_find_min(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::host_find_min<T>");

    return host_find_min_impl<T>(x.get_vec(), x.get_size());
}

// Find maximum
template<typename T>
T linalg::host_find_max(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::host_find_max<T>");

    return host_find_max_impl<T>(x.get_vec(), x.get_size());
}

// Exclusive scan
template <typename T>
void linalg::host_exclusive_scan(vector<T>& x)
{
    ROUTINE_TRACE("linalg::host_exclusive_scan<T>");

    host_exclusive_scan_impl<T>(x.get_vec(), x.get_size());
}

template int32_t linalg::host_find_min<int32_t>(const vector<int32_t>& x);
template int64_t linalg::host_find_min<int64_t>(const vector<int64_t>& x);
template double linalg::host_find_min<double>(const vector<double>& x);

template int32_t linalg::host_find_max<int32_t>(const vector<int32_t>& x);
template int64_t linalg::host_find_max<int64_t>(const vector<int64_t>& x);
template double linalg::host_find_max<double>(const vector<double>& x);

template void linalg::host_exclusive_scan<int32_t>(vector<int32_t>& x);
template void linalg::host_exclusive_scan<int64_t>(vector<int64_t>& x);
template void linalg::host_exclusive_scan<double>(vector<double>& x);
