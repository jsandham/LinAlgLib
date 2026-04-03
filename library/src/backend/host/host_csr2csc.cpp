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

#include "host_csr2csc.h"
#include "../../trace.h"

void linalg::host_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::host_transpose_matrix");
    transposeA.resize(A.get_n(), A.get_m(), A.get_nnz());
    int*    csr_row_ptr_T = transposeA.get_row_ptr();
    int*    csr_col_ind_T = transposeA.get_col_ind();
    double* csr_val_T     = transposeA.get_val();
    for(size_t i = 0; i < transposeA.get_m() + 1; i++)
    {
        csr_row_ptr_T[i] = 0;
    }
    for(size_t i = 0; i < transposeA.get_nnz(); i++)
    {
        csr_col_ind_T[i] = -1;
    }
    const int*    csr_row_ptr_A = A.get_row_ptr();
    const int*    csr_col_ind_A = A.get_col_ind();
    const double* csr_val_A     = A.get_val();
    for(int i = 0; i < A.get_m(); i++)
    {
        for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
        {
            csr_row_ptr_T[csr_col_ind_A[j] + 1]++;
        }
    }
    for(int i = 0; i < transposeA.get_m(); i++)
    {
        csr_row_ptr_T[i + 1] += csr_row_ptr_T[i];
    }
    for(int i = 0; i < A.get_m(); i++)
    {
        for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
        {
            int col = csr_col_ind_A[j];
            for(int k = csr_row_ptr_T[col]; k < csr_row_ptr_T[col + 1]; k++)
            {
                if(csr_col_ind_T[k] == -1)
                {
                    csr_col_ind_T[k] = i;
                    csr_val_T[k]     = csr_val_A[j];
                    break;
                }
            }
        }
    }
}
