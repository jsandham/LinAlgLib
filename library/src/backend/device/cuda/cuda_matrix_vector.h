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
#ifndef CUDA_MATRIX_VECTOR_H
#define CUDA_MATRIX_VECTOR_H

#include "linalg_enums.h"

namespace linalg
{
    void cuda_matrix_vector_product(int           m,
                                    int           n,
                                    int           nnz,
                                    const int*    csr_row_ptr,
                                    const int*    csr_col_ind,
                                    const double* csr_val,
                                    const double* x,
                                    double*       y);
    void cuda_compute_residual(int           m,
                               int           n,
                               int           nnz,
                               const int*    csr_row_ptr,
                               const int*    csr_col_ind,
                               const double* csr_val,
                               const double* x,
                               const double* b,
                               double*       res);

    struct csrmv_descr;

    void allocate_csrmv_cuda_data(csrmv_descr* descr);
    void free_csrmv_cuda_data(csrmv_descr* descr);

    void cuda_csrmv_analysis(int             m,
                             int             n,
                             int             nnz,
                             const int*      csr_row_ptr,
                             const int*      csr_col_ind,
                             const double*   csr_val,
                             csrmv_algorithm alg,
                             csrmv_descr*    descr);
    void cuda_csrmv_solve(int                m,
                          int                n,
                          int                nnz,
                          double             alpha,
                          const int*         csr_row_ptr,
                          const int*         csr_col_ind,
                          const double*      csr_val,
                          const double*      x,
                          double             beta,
                          double*            y,
                          csrmv_algorithm    alg,
                          const csrmv_descr* descr);
}

#endif
