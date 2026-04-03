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

#include <cassert>
#include <cmath>

#include "../../trace.h"

#include "host_csrtrsv.h"

namespace linalg
{
    template <typename T>
    static void host_forward_solve_impl(const int* csr_row_ptr,
                                        const int* csr_col_ind,
                                        const T*   csr_val,
                                        const T*   b,
                                        T*         x,
                                        int        n,
                                        bool       unit_diag)
    {
        ROUTINE_TRACE("host_forward_solve_impl");
        for(int i = 0; i < n; i++)
        {
            T diag_val = 1.0;
            T sum      = 0.0;
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                int col = csr_col_ind[j];
                if(col < i)
                {
                    sum = std::fma(csr_val[j], x[col], sum);
                }
                else if(!unit_diag && col == i)
                {
                    diag_val = csr_val[j];
                }
                else
                {
                    break;
                }
            }
            x[i] = (b[i] - sum) / diag_val;
        }
    }

    template <typename T>
    static void host_backward_solve_impl(const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         const T*   b,
                                         T*         x,
                                         int        n,
                                         bool       unit_diag)
    {
        ROUTINE_TRACE("host_backward_solve_impl");
        for(int i = n - 1; i >= 0; i--)
        {
            T diag_val = 1.0;
            x[i]       = b[i];
            for(int j = csr_row_ptr[i + 1] - 1; j >= csr_row_ptr[i]; j--)
            {
                int col = csr_col_ind[j];
                if(col > i)
                {
                    x[i] -= csr_val[j] * x[col];
                }
                else if(!unit_diag && col == i)
                {
                    diag_val = csr_val[j];
                }
            }
            x[i] /= diag_val;
        }
    }
}

void linalg::host_csrtrsv_analysis(const csr_matrix& A,
                                   triangular_type   tri_type,
                                   diagonal_type     diag_type,
                                   csrtrsv_descr*    descr)
{
    ROUTINE_TRACE("linalg::host_csrtrsv_analysis");
}

void linalg::host_csrtrsv_solve(const csr_matrix&     A,
                                const vector<double>& b,
                                vector<double>&       x,
                                double                alpha,
                                triangular_type       tri_type,
                                diagonal_type         diag_type,
                                const csrtrsv_descr*  descr)
{
    ROUTINE_TRACE("linalg::host_csrtrsv_solve");
    if(tri_type == triangular_type::upper)
    {
        host_backward_solve_impl(A.get_row_ptr(),
                                 A.get_col_ind(),
                                 A.get_val(),
                                 b.get_vec(),
                                 x.get_vec(),
                                 A.get_m(),
                                 diag_type == diagonal_type::unit);
    }
    else if(tri_type == triangular_type::lower)
    {
        host_forward_solve_impl(A.get_row_ptr(),
                                A.get_col_ind(),
                                A.get_val(),
                                b.get_vec(),
                                x.get_vec(),
                                A.get_m(),
                                diag_type == diagonal_type::unit);
    }
}
