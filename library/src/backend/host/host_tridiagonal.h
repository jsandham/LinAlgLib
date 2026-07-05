//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#ifndef HOST_TRIDIAGONAL_H
#define HOST_TRIDIAGONAL_H

#include "../../../include/direct_solvers/tridiagonal/tridiagonal.h"
#include "vector.h"

namespace linalg
{
    template <typename T>
    void host_partial_pivoting_algorithm(int               m,
                                         int               n,
                                         const vector<T>&  lower_diag,
                                         const vector<T>&  main_diag,
                                         const vector<T>&  upper_diag,
                                         const vector<T>&  rhs,
                                         vector<T>&        solution,
                                         pivoting_data<T>& pivot_data);
    template <typename T>
    void host_non_pivoting_algorithm(int                   m,
                                     int                   n,
                                     const vector<T>&      lower_diag,
                                     const vector<T>&      main_diag,
                                     const vector<T>&      upper_diag,
                                     const vector<T>&      rhs,
                                     vector<T>&            solution,
                                     non_pivoting_data<T>& non_pivot_data);
}

#endif
