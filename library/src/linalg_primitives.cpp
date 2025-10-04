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

#include "../include/linalg_primitives.h"
#include <assert.h>
#include <iostream>

#include "trace.h"
#include "utility.h"

#include "backend/device/device_primitives.h"
#include "backend/host/host_primitives.h"

// Find minimum
template<typename T>
T linalg::find_min(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::find_min<T>");

    return backend_dispatch("linalg::find_min<T>", host_find_min<T>, device_find_min<T>, x);
}

// Find maximum
template<typename T>
T linalg::find_max(const vector<T>& x)
{
    ROUTINE_TRACE("linalg::find_max<T>");

    return backend_dispatch("linalg::find_max<T>", host_find_max<T>, device_find_max<T>, x);
}

// Exclusive scan
template <typename T>
void linalg::exclusive_scan(vector<T>& x)
{
    ROUTINE_TRACE("linalg::exclusive_scan<T>");

    backend_dispatch("linalg::exclusive_scan<T>", host_exclusive_scan<T>, device_exclusive_scan<T>, x);
}

template int32_t linalg::find_min<int32_t>(const vector<int32_t>& x);
template int64_t linalg::find_min<int64_t>(const vector<int64_t>& x);
template double linalg::find_min<double>(const vector<double>& x);

template int32_t linalg::find_max<int32_t>(const vector<int32_t>& x);
template int64_t linalg::find_max<int64_t>(const vector<int64_t>& x);
template double linalg::find_max<double>(const vector<double>& x);

template void linalg::exclusive_scan<int32_t>(vector<int32_t>& x);
template void linalg::exclusive_scan<int64_t>(vector<int64_t>& x);
template void linalg::exclusive_scan<double>(vector<double>& x);