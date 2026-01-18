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

#include "linalg_enums.h"

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

    void cuda_extract_lower_triangular_nnz(int        m_A,
                                           int        n_A,
                                           int        nnz_A,
                                           const int* csr_row_ptr_A,
                                           const int* csr_col_ind_A,
                                           int*       csr_row_ptr_L,
                                           int*       nnz_L);
    void cuda_extract_lower_triangular(int           m_A,
                                       int           n_A,
                                       int           nnz_A,
                                       const int*    csr_row_ptr_A,
                                       const int*    csr_col_ind_A,
                                       const double* csr_val_A,
                                       int           m_L,
                                       int           n_L,
                                       int           nnz_L,
                                       int*          csr_row_ptr_L,
                                       int*          csr_col_ind_L,
                                       double*       csr_val_L);

    void cuda_extract_upper_triangular_nnz(int        m_A,
                                           int        n_A,
                                           int        nnz_A,
                                           const int* csr_row_ptr_A,
                                           const int* csr_col_ind_A,
                                           int*       csr_row_ptr_U,
                                           int*       nnz_U);
    void cuda_extract_upper_triangular(int           m_A,
                                       int           n_A,
                                       int           nnz_A,
                                       const int*    csr_row_ptr_A,
                                       const int*    csr_col_ind_A,
                                       const double* csr_val_A,
                                       int           m_U,
                                       int           n_U,
                                       int           nnz_U,
                                       int*          csr_row_ptr_U,
                                       int*          csr_col_ind_U,
                                       double*       csr_val_U);

    void cuda_scale_diagonal(
        const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, double scalar);

    void cuda_scale_by_inverse_diagonal(
        const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, const double* diag);

    double cuda_norm_inf(const double* array, int size);
    void   cuda_jacobi_solve(const double* rhs, const double* diag, double* x, size_t size);
    void   cuda_ssor_fill_lower_precond(int           m_A,
                                        int           n_A,
                                        int           nnz_A,
                                        const int*    csr_row_ptr_A,
                                        const int*    csr_col_ind_A,
                                        const double* csr_val_A,
                                        int           m_L,
                                        int           n_L,
                                        int           nnz_L,
                                        const int*    csr_row_ptr_L,
                                        int*          csr_col_ind_L,
                                        double*       csr_val_L,
                                        double        omega);
    void   cuda_ssor_fill_upper_precond(int           m_A,
                                        int           n_A,
                                        int           nnz_A,
                                        const int*    csr_row_ptr_A,
                                        const int*    csr_col_ind_A,
                                        const double* csr_val_A,
                                        int           m_U,
                                        int           n_U,
                                        int           nnz_U,
                                        const int*    csr_row_ptr_U,
                                        int*          csr_col_ind_U,
                                        double*       csr_val_U,
                                        double        omega);

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

    struct csrtrsv_descr;

    void allocate_csrtrsv_cuda_data(csrtrsv_descr* descr);
    void free_csrtrsv_cuda_data(csrtrsv_descr* descr);

    void cuda_csrtrsv_analysis(int             m,
                               int             n,
                               int             nnz,
                               const int*      csr_row_ptr,
                               const int*      csr_col_ind,
                               const double*   csr_val,
                               triangular_type tri_type,
                               diagonal_type   diag_type,
                               csrtrsv_descr*  descr);
    void cuda_csrtrsv_solve(int                  m,
                            int                  n,
                            int                  nnz,
                            double               alpha,
                            const int*           csr_row_ptr,
                            const int*           csr_col_ind,
                            const double*        csr_val,
                            const double*        b,
                            double*              x,
                            triangular_type      tri_type,
                            diagonal_type        diag_type,
                            const csrtrsv_descr* descr);

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

    struct csrgeam_descr;

    void allocate_csrgeam_cuda_data(csrgeam_descr* descr);
    void free_csrgeam_cuda_data(csrgeam_descr* descr);

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

    void cuda_csrgeam_solve(int                  m,
                            int                  n,
                            int                  nnz_A,
                            int                  nnz_B,
                            int                  nnz_C,
                            const csrgeam_descr* descr,
                            double               alpha,
                            const int*           csr_row_ptr_A,
                            const int*           csr_col_ind_A,
                            const double*        csr_val_A,
                            double               beta,
                            const int*           csr_row_ptr_B,
                            const int*           csr_col_ind_B,
                            const double*        csr_val_B,
                            const int*           csr_row_ptr_C,
                            int*                 csr_col_ind_C,
                            double*              csr_val_C);

    struct csrgemm_descr;

    void allocate_csrgemm_cuda_data(csrgemm_descr* descr);
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

    struct csric0_descr;

    void allocate_csric0_cuda_data(csric0_descr* descr);
    void free_csric0_cuda_data(csric0_descr* descr);

    void cuda_csric0_analysis(int           m,
                              int           n,
                              int           nnz,
                              const int*    csr_row_ptr,
                              const int*    csr_col_ind,
                              const double* csr_val,
                              csric0_descr* descr);

    void cuda_csric0_compute(int                 m,
                             int                 n,
                             int                 nnz,
                             const int*          csr_row_ptr,
                             const int*          csr_col_ind,
                             double*             csr_val,
                             const csric0_descr* descr);

}

#endif
