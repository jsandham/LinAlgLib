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

#ifndef CSRMV_DESCR_INTERNAL_H
#define CSRMV_DESCR_INTERNAL_H

#include <array>

namespace linalg
{
    struct csrmv_descr
    {
        // LRB algorithm
        std::array<int, 32> hbin_count; // how many rows belong to each bin

        int* bin_count; // desvice array of size 32, how many rows belong to each bin
        int* bin_start_ptr; // device array of size (32 + 1)
        int* row_index_in_bin; // device array of size m         row_indices_perm?
        int* row_index_in_bin_sorted; // device array of size m  row_indices?
    };
}

#endif // CSRMV_DESCR_INTERNAL_H
