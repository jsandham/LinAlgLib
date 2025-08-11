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

namespace linalg
{
    void   device_axpy_impl(int size, double alpha, const double* x, double* y);
    void   device_axpby_impl(int size, double alpha, const double* x, double beta, double* y);
    void   device_axpbypgz_impl(int           size,
                                double        alpha,
                                const double* x,
                                double        beta,
                                const double* y,
                                double        gamma,
                                double*       z);
    void   device_matrix_vector_product_impl(int           m,
                                             int           n,
                                             int           nnz,
                                             const int*    csr_row_ptr,
                                             const int*    csr_col_ind,
                                             const double* csr_val,
                                             const double* x,
                                             double*       y);
    double device_dot_product_impl(const double* x, const double* y, int size);
    void   device_compute_residual_impl(int           m,
                                        int           n,
                                        int           nnz,
                                        const int*    csr_row_ptr,
                                        const int*    csr_col_ind,
                                        const double* csr_val,
                                        const double* x,
                                        const double* b,
                                        double*       res);
    void   device_exclusive_scan_impl(double* x, int n);
    void   device_extract_diagonal_impl(int           m,
                                        int           n,
                                        int           nnz,
                                        const int*    csr_row_ptr,
                                        const int*    csr_col_ind,
                                        const double* csr_val,
                                        double*       d);
    double device_norm_inf_impl(const double* array, int size);
    void   device_jacobi_solve_impl(const double* rhs, const double* diag, double* x, size_t size);
    void   device_csrmv_impl(int           m,
                             int           n,
                             int           nnz,
                             double        alpha,
                             const int*    csr_row_ptr,
                             const int*    csr_col_ind,
                             const double* csr_val,
                             const double* x,
                             double        beta,
                             double*       y);
    void   device_csrgemm_nnz_impl(int        m,
                                   int        n,
                                   int        k,
                                   int        nnz_A,
                                   int        nnz_B,
                                   int        nnz_D,
                                   double     alpha,
                                   const int* csr_row_ptr_A,
                                   const int* csr_col_ind_A,
                                   const int* csr_row_ptr_B,
                                   const int* csr_col_ind_B,
                                   double     beta,
                                   const int* csr_row_ptr_D,
                                   const int* csr_col_ind_D,
                                   int*       csr_row_ptr_C,
                                   int*       nnz_C);

    void device_csrgemm_impl(int           m,
                             int           n,
                             int           k,
                             int           nnz_A,
                             int           nnz_B,
                             int           nnz_D,
                             double        alpha,
                             const int*    csr_row_ptr_A,
                             const int*    csr_col_ind_A,
                             const double* csr_val_A,
                             const int*    csr_row_ptr_B,
                             const int*    csr_col_ind_B,
                             const double* csr_val_B,
                             double        beta,
                             const int*    csr_row_ptr_D,
                             const int*    csr_col_ind_D,
                             const double* csr_val_D,
                             const int*    csr_row_ptr_C,
                             int*          csr_col_ind_C,
                             double*       csr_val_C);
    void device_csrgeam_nnz_impl(int        m,
                                 int        n,
                                 int        nnz_A,
                                 int        nnz_B,
                                 double     alpha,
                                 const int* csr_row_ptr_A,
                                 const int* csr_col_ind_A,
                                 double     beta,
                                 const int* csr_row_ptr_B,
                                 const int* csr_col_ind_B,
                                 int*       csr_row_ptr_C,
                                 int*       nnz_C);

    void device_csrgeam_impl(int           m,
                             int           n,
                             int           nnz_A,
                             int           nnz_B,
                             double        alpha,
                             const int*    csr_row_ptr_A,
                             const int*    csr_col_ind_A,
                             const double* csr_val_A,
                             double        beta,
                             const int*    csr_row_ptr_B,
                             const int*    csr_col_ind_B,
                             const double* csr_val_B,
                             const int*    csr_row_ptr_C,
                             int*          csr_col_ind_C,
                             double*       csr_val_C);

    void device_csrilu0_impl(int        m,
                             int        n,
                             int        nnz,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             double*    csr_val,
                             int*       structural_zero,
                             int*       numeric_zero);
    void device_csric0_impl(int        m,
                            int        n,
                            int        nnz,
                            const int* csr_row_ptr,
                            const int* csr_col_ind,
                            double*    csr_val,
                            int*       structural_zero,
                            int*       numeric_zero);

    void device_forward_solve_impl(const int*    csr_row_ptr,
                                   const int*    csr_col_ind,
                                   const double* csr_val,
                                   const double* b,
                                   double*       x,
                                   int           n,
                                   bool          unit_diag);

    void device_backward_solve_impl(const int*    csr_row_ptr,
                                    const int*    csr_col_ind,
                                    const double* csr_val,
                                    const double* b,
                                    double*       x,
                                    int           n,
                                    bool          unit_diag);
}