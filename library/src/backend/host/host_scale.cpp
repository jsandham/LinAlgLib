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

#include "host_scale.h"

#include "../../trace.h"

namespace linalg
{
    template <typename T>
    static void host_scale_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int n, T scalar)
    {
        ROUTINE_TRACE("host_scale_diagonal_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                if(csr_col_ind[j] == i)
                {
                    csr_val[j] *= scalar;
                    break;
                }
            }
        }
    }

    template <typename T>
    static void host_scale_by_inverse_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int n, const T* diag)
    {
        ROUTINE_TRACE("host_scale_by_inverse_diagonal_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                csr_val[j] *= (1.0 / diag[csr_col_ind[j]]);
            }
        }
    }
}

void linalg::host_scale_diagonal(csr_matrix& A, double scalar)
{
    ROUTINE_TRACE("linalg::host_scale_diagonal");
    host_scale_diagonal_impl(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), scalar);
}

void linalg::host_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag)
{
    ROUTINE_TRACE("linalg::host_scale_by_inverse_diagonal");
    host_scale_by_inverse_diagonal_impl(
        A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), diag.get_vec());
}
