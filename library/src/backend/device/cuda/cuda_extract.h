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
#ifndef CUDA_EXTRACT_H
#define CUDA_EXTRACT_H

namespace linalg
{
    template <typename T>
    void cuda_extract_diagonal(int        m,
                               int        n,
                               int        nnz,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               const T*   csr_val,
                               T*         d);

    void cuda_extract_lower_triangular_nnz(int        m_A,
                                           int        n_A,
                                           int        nnz_A,
                                           const int* csr_row_ptr_A,
                                           const int* csr_col_ind_A,
                                           int*       csr_row_ptr_L,
                                           int*       nnz_L);

    template <typename T>
    void cuda_extract_lower_triangular(int        m_A,
                                       int        n_A,
                                       int        nnz_A,
                                       const int* csr_row_ptr_A,
                                       const int* csr_col_ind_A,
                                       const T*   csr_val_A,
                                       int        m_L,
                                       int        n_L,
                                       int        nnz_L,
                                       int*       csr_row_ptr_L,
                                       int*       csr_col_ind_L,
                                       T*         csr_val_L);

    void cuda_extract_upper_triangular_nnz(int        m_A,
                                           int        n_A,
                                           int        nnz_A,
                                           const int* csr_row_ptr_A,
                                           const int* csr_col_ind_A,
                                           int*       csr_row_ptr_U,
                                           int*       nnz_U);
    template <typename T>
    void cuda_extract_upper_triangular(int        m_A,
                                       int        n_A,
                                       int        nnz_A,
                                       const int* csr_row_ptr_A,
                                       const int* csr_col_ind_A,
                                       const T*   csr_val_A,
                                       int        m_U,
                                       int        n_U,
                                       int        nnz_U,
                                       int*       csr_row_ptr_U,
                                       int*       csr_col_ind_U,
                                       T*         csr_val_U);
}

#endif
