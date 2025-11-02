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
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

namespace linalg
{
    void   cuda_axpy(int size, double alpha, const double* x, double* y);
    void   cuda_axpby(int size, double alpha, const double* x, double beta, double* y);
    void   cuda_axpbypgz(int           size,
                         double        alpha,
                         const double* x,
                         double        beta,
                         const double* y,
                         double        gamma,
                         double*       z);
    void   cuda_matrix_vector_product(int           m,
                                      int           n,
                                      int           nnz,
                                      const int*    csr_row_ptr,
                                      const int*    csr_col_ind,
                                      const double* csr_val,
                                      const double* x,
                                      double*       y);
    double cuda_dot_product(const double* x, const double* y, int size);
    void   cuda_compute_residual(int           m,
                                 int           n,
                                 int           nnz,
                                 const int*    csr_row_ptr,
                                 const int*    csr_col_ind,
                                 const double* csr_val,
                                 const double* x,
                                 const double* b,
                                 double*       res);
    void   cuda_extract_diagonal(int           m,
                                 int           n,
                                 int           nnz,
                                 const int*    csr_row_ptr,
                                 const int*    csr_col_ind,
                                 const double* csr_val,
                                 double*       d);
    double cuda_norm_inf(const double* array, int size);
    void   cuda_jacobi_solve(const double* rhs, const double* diag, double* x, size_t size);
    void   cuda_csrmv(int           m,
                      int           n,
                      int           nnz,
                      double        alpha,
                      const int*    csr_row_ptr,
                      const int*    csr_col_ind,
                      const double* csr_val,
                      const double* x,
                      double        beta,
                      double*       y);

    struct csrgemm_descr;

    void cuda_create_csrgemm_descr(csrgemm_descr** descr);
    void cuda_destroy_csrgemm_descr(csrgemm_descr* descr);

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

    void cuda_csrgemm(int            m,
                      int            n,
                      int            k,
                      int            nnz_A,
                      int            nnz_B,
                      int            nnz_D,
                      int            nnz_C,
                      csrgemm_descr* descr,
                      double         alpha,
                      const int*     csr_row_ptr_A,
                      const int*     csr_col_ind_A,
                      const double*  csr_val_A,
                      const int*     csr_row_ptr_B,
                      const int*     csr_col_ind_B,
                      const double*  csr_val_B,
                      double         beta,
                      const int*     csr_row_ptr_D,
                      const int*     csr_col_ind_D,
                      const double*  csr_val_D,
                      const int*     csr_row_ptr_C,
                      int*           csr_col_ind_C,
                      double*        csr_val_C);

    struct csrgeam_descr;

    void cuda_create_csrgeam_descr(csrgeam_descr** descr);
    void cuda_destroy_csrgeam_descr(csrgeam_descr* descr);

    void cuda_csrgeam_nnz(int            m,
                          int            n,
                          int            nnz_A,
                          int            nnz_B,
                          csrgeam_descr* descr,
                          double         alpha,
                          const int*     csr_row_ptr_A,
                          const int*     csr_col_ind_A,
                          double         beta,
                          const int*     csr_row_ptr_B,
                          const int*     csr_col_ind_B,
                          int*           csr_row_ptr_C,
                          int*           nnz_C);

    void cuda_csrgeam(int            m,
                      int            n,
                      int            nnz_A,
                      int            nnz_B,
                      int            nnz_C,
                      csrgeam_descr* descr,
                      double         alpha,
                      const int*     csr_row_ptr_A,
                      const int*     csr_col_ind_A,
                      const double*  csr_val_A,
                      double         beta,
                      const int*     csr_row_ptr_B,
                      const int*     csr_col_ind_B,
                      const double*  csr_val_B,
                      const int*     csr_row_ptr_C,
                      int*           csr_col_ind_C,
                      double*        csr_val_C);

    void cuda_csrilu0(int        m,
                      int        n,
                      int        nnz,
                      const int* csr_row_ptr,
                      const int* csr_col_ind,
                      double*    csr_val,
                      int*       structural_zero,
                      int*       numeric_zero);
    void cuda_csric0(int        m,
                     int        n,
                     int        nnz,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     double*    csr_val,
                     int*       structural_zero,
                     int*       numeric_zero);

    void cuda_forward_solve(const int*    csr_row_ptr,
                            const int*    csr_col_ind,
                            const double* csr_val,
                            const double* b,
                            double*       x,
                            int           n,
                            bool          unit_diag);

    void cuda_backward_solve(const int*    csr_row_ptr,
                             const int*    csr_col_ind,
                             const double* csr_val,
                             const double* b,
                             double*       x,
                             int           n,
                             bool          unit_diag);

    void cuda_csr2csc_buffer_size(int           m,
                                  int           n,
                                  int           nnz,
                                  const int*    csr_row_ptr,
                                  const int*    csr_col_ind,
                                  const double* csr_val,
                                  size_t*       buffer_size);

    void cuda_csr2csc(int           m,
                      int           n,
                      int           nnz,
                      const int*    csr_row_ptr,
                      const int*    csr_col_ind,
                      const double* csr_val,
                      int*          csc_col_ptr,
                      int*          csc_row_ind,
                      double*       csc_val,
                      void*         buffer);
}

#endif