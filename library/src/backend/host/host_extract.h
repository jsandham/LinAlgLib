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

#ifndef HOST_EXTRACT_H
#define HOST_EXTRACT_H

#include "csr_matrix.h"
#include "vector.h"

namespace linalg
{
    void host_diagonal(const csr_matrix& A, vector<double>& d);
    void host_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L);
    void host_extract_lower_triangular(const csr_matrix& A, csr_matrix& L);
    void host_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U);
    void host_extract_upper_triangular(const csr_matrix& A, csr_matrix& U);
}

#endif
