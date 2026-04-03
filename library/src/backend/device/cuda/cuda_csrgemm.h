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
#ifndef CUDA_CSRGEMM_H
#define CUDA_CSRGEMM_H

namespace linalg
{
    struct csrgemm_descr;

    void free_csrgemm_cuda_data(csrgemm_descr* descr);

    void cuda_csrgemm_nnz(int            m,
                          int            n,
                          int            k,
                          int            nnz_A,
                          int            nnz_B,
                          int            nnz_D,
                          csrgemm_descr* descr,
                          double         alpha,
                          const int*     csr_row_ptr_A,
                          const int*     csr_col_ind_A,
                          const int*     csr_row_ptr_B,
                          const int*     csr_col_ind_B,
                          double         beta,
                          const int*     csr_row_ptr_D,
                          const int*     csr_col_ind_D,
                          int*           csr_row_ptr_C,
                          int*           nnz_C);

    void cuda_csrgemm_solve(int                  m,
                            int                  n,
                            int                  k,
                            int                  nnz_A,
                            int                  nnz_B,
                            int                  nnz_D,
                            int                  nnz_C,
                            const csrgemm_descr* descr,
                            double               alpha,
                            const int*           csr_row_ptr_A,
                            const int*           csr_col_ind_A,
                            const double*        csr_val_A,
                            const int*           csr_row_ptr_B,
                            const int*           csr_col_ind_B,
                            const double*        csr_val_B,
                            double               beta,
                            const int*           csr_row_ptr_D,
                            const int*           csr_col_ind_D,
                            const double*        csr_val_D,
                            const int*           csr_row_ptr_C,
                            int*                 csr_col_ind_C,
                            double*              csr_val_C);
}

#endif
