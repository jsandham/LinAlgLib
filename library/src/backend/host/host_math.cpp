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

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../trace.h"
#include "host_math.h"

namespace linalg
{
    //-------------------------------------------------------------------------------
    // Compute y = alpha * x + y
    //-------------------------------------------------------------------------------
    static void host_axpy_impl(int n, double alpha, const double* x, double* y)
    {
        ROUTINE_TRACE("host_axpy_impl");

        if(alpha == 1.0)
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = x[i] + y[i];
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = alpha * x[i] + y[i];
            }
        }
    }

    //-------------------------------------------------------------------------------
    // Compute y = alpha * x + beta * y
    //-------------------------------------------------------------------------------
    static void host_axpby_impl(int n, double alpha, const double* x, double beta, double* y)
    {
        ROUTINE_TRACE("host_axpby_impl");

        if(alpha == 1.0)
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = x[i] + beta * y[i];
            }
        }
        else if(alpha == 0.0)
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = beta * y[i];
            }
        }
        else if(beta == 1.0)
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = alpha * x[i] + y[i];
            }
        }
        else if(beta == 0.0)
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = alpha * x[i];
            }
        }
        else
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < n; i++)
            {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }
    }

    //-------------------------------------------------------------------------------
    // Compute z = alpha * x + beta * y + gamma * z
    //-------------------------------------------------------------------------------
    static void host_axpbypgz_impl(
        int n, double alpha, const double* x, double beta, const double* y, double gamma, double* z)
    {
        ROUTINE_TRACE("host_axpbypgz_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            z[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
        }
    }

    //-------------------------------------------------------------------------------
    // sparse matrix-vector product y = A*x
    //-------------------------------------------------------------------------------
    static void host_matrix_vector_product_impl(const int*    csr_row_ptr,
                                                const int*    csr_col_ind,
                                                const double* csr_val,
                                                const double* x,
                                                double*       y,
                                                int           n)
    {
        ROUTINE_TRACE("host_matrix_vector_product_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            double s = 0.0;
            for(int j = row_start; j < row_end; j++)
            {
                s += csr_val[j] * x[csr_col_ind[j]];
            }

            y[i] = s;
        }
    }

    //-------------------------------------------------------------------------------
    // dot product z = x*y
    //-------------------------------------------------------------------------------
    static double host_dot_product_impl(const double* x, const double* y, int n)
    {
        ROUTINE_TRACE("host_dot_product_impl");

        double dot_prod = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : dot_prod)
#endif
        for(int i = 0; i < n; i++)
        {
            dot_prod += x[i] * y[i];
        }

        return dot_prod;
    }

    //-------------------------------------------------------------------------------
    // Compute residual res = b - A * x
    //-------------------------------------------------------------------------------
    static void host_compute_residual_impl(const int*    csr_row_ptr,
                                           const int*    csr_col_ind,
                                           const double* csr_val,
                                           const double* x,
                                           const double* b,
                                           double*       res,
                                           int           n)
    {
        ROUTINE_TRACE("host_compute_residual_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            double s = 0.0;
            for(int j = row_start; j < row_end; j++)
            {
                s += csr_val[j] * x[csr_col_ind[j]];
            }

            res[i] = b[i] - s;
        }
    }

    //-------------------------------------------------------------------------------
    // exclusive scan
    //-------------------------------------------------------------------------------
    static void host_exclusive_scan_impl(double* x, int n)
    {
        ROUTINE_TRACE("host_exclusive_scan_impl");

        if(n > 0)
        {
            x[0] = 0;

            for(int i = 0; i < n - 1; i++)
            {
                x[i + 1] += x[i];
            }
        }
    }

    //-------------------------------------------------------------------------------
    // diagonal d = diag(A)
    //-------------------------------------------------------------------------------
    static void host_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* d, int n)
    {
        ROUTINE_TRACE("host_diagonal_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            for(int j = row_start; j < row_end; j++)
            {
                if(csr_col_ind[j] == i)
                {
                    d[i] = csr_val[j];
                    break;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    // infinity norm
    //-------------------------------------------------------------------------------
    static double host_norm_inf_impl(const double* array, int n)
    {
        ROUTINE_TRACE("host_norm_inf_impl");

        double norm = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(max : norm)
#endif
        for(int i = 0; i < n; i++)
        {
            norm = std::max(std::abs(array[i]), norm);
        }

        return norm;
    }

    //-------------------------------------------------------------------------------
    // jacobi solve
    //-------------------------------------------------------------------------------
    static void host_jacobi_solve_impl(const double* rhs, const double* diag, double* x, size_t n)
    {
        ROUTINE_TRACE("host_jacobi_solve_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < n; i++)
        {
            x[i] = rhs[i] / diag[i];
        }
    }

    //-------------------------------------------------------------------------------
    // Compute y = alpha * A * x + beta * y
    //-------------------------------------------------------------------------------
    static void host_csrmv_impl(int           m,
                                int           n,
                                int           nnz,
                                double        alpha,
                                const int*    csr_row_ptr,
                                const int*    csr_col_ind,
                                const double* csr_val,
                                const double* x,
                                double        beta,
                                double*       y)
    {
        ROUTINE_TRACE("host_csrmv");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m; i++)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            double s = 0.0;
            for(int j = row_start; j < row_end; j++)
            {
                s += csr_val[j] * x[csr_col_ind[j]];
            }

            if(beta == 0.0)
            {
                y[i] = alpha * s;
            }
            else
            {
                y[i] = alpha * s + beta * y[i];
            }
        }
    }

    //-------------------------------------------------------------------------------
    // Compute C = alpha * A * B + beta * D
    //-------------------------------------------------------------------------------
    static void host_csrgemm_nnz_impl(int        m,
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
                                      int*       nnz_C)
    {
        ROUTINE_TRACE("host_csrgemm_nnz");

        std::vector<int> nnz(n, -1);

        // A is mxk, B is kxn, and C is mxn
        for(int i = 0; i < m + 1; i++)
        {
            csr_row_ptr_C[i] = 0;
        }

        for(int i = 0; i < m; ++i)
        {
            int row_begin_A = csr_row_ptr_A[i];
            int row_end_A   = csr_row_ptr_A[i + 1];

            for(int j = row_begin_A; j < row_end_A; j++)
            {
                int col_A = csr_col_ind_A[j];

                int row_begin_B = csr_row_ptr_B[col_A];
                int row_end_B   = csr_row_ptr_B[col_A + 1];

                for(int p = row_begin_B; p < row_end_B; p++)
                {
                    int col_B = csr_col_ind_B[p];

                    if(nnz[col_B] != i)
                    {
                        nnz[col_B] = i;
                        csr_row_ptr_C[i + 1]++;
                    }
                }
            }

            if(beta != 0.0)
            {
                int row_begin_D = csr_row_ptr_D[i];
                int row_end_D   = csr_row_ptr_D[i + 1];

                for(int j = row_begin_D; j < row_end_D; j++)
                {
                    int col_D = csr_col_ind_D[j];

                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        csr_row_ptr_C[i + 1]++;
                    }
                }
            }
        }

        for(int i = 0; i < m; i++)
        {
            csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
        }

        *nnz_C = csr_row_ptr_C[m];
    }

    static void host_csrgemm_impl(int           m,
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
                                  double*       csr_val_C)
    {
        ROUTINE_TRACE("host_csrgemm_impl");

        std::vector<int> nnzs(n, -1);

        for(int i = 0; i < m; i++)
        {
            int row_begin_C = csr_row_ptr_C[i];
            int row_end_C   = row_begin_C;

            int row_begin_A = csr_row_ptr_A[i];
            int row_end_A   = csr_row_ptr_A[i + 1];

            for(int j = row_begin_A; j < row_end_A; j++)
            {
                int    col_A = csr_col_ind_A[j];
                double val_A = alpha * csr_val_A[j];

                int row_begin_B = csr_row_ptr_B[col_A];
                int row_end_B   = csr_row_ptr_B[col_A + 1];

                for(int p = row_begin_B; p < row_end_B; p++)
                {
                    int    col_B = csr_col_ind_B[p];
                    double val_B = csr_val_B[p];

                    if(nnzs[col_B] < row_begin_C)
                    {
                        nnzs[col_B]              = row_end_C;
                        csr_col_ind_C[row_end_C] = col_B;
                        csr_val_C[row_end_C]     = val_A * val_B;
                        row_end_C++;
                    }
                    else
                    {
                        csr_val_C[nnzs[col_B]] += val_A * val_B;
                    }
                }
            }

            if(beta != 0.0)
            {
                int row_begin_D = csr_row_ptr_D[i];
                int row_end_D   = csr_row_ptr_D[i + 1];

                for(int j = row_begin_D; j < row_end_D; j++)
                {
                    int    col_D = csr_col_ind_D[j];
                    double val_D = beta * csr_val_D[j];

                    // Check if a new nnz is generated or if the value is added
                    if(nnzs[col_D] < row_begin_C)
                    {
                        nnzs[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D;
                        csr_val_C[row_end_C]     = val_D;
                        row_end_C++;
                    }
                    else
                    {
                        csr_val_C[nnzs[col_D]] += val_D;
                    }
                }
            }
        }

        int nnz = csr_row_ptr_C[m];

        std::vector<int>    cols(nnz);
        std::vector<double> vals(nnz);

        memcpy(cols.data(), csr_col_ind_C, sizeof(int) * nnz);
        memcpy(vals.data(), csr_val_C, sizeof(double) * nnz);

        for(int i = 0; i < m; i++)
        {
            int row_begin = csr_row_ptr_C[i];
            int row_end   = csr_row_ptr_C[i + 1];
            int row_nnz   = row_end - row_begin;

            std::vector<int> perm(row_nnz);
            for(int j = 0; j < row_nnz; j++)
            {
                perm[j] = j;
            }

            int*    col_entry = cols.data() + row_begin;
            double* val_entry = vals.data() + row_begin;

            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                return col_entry[a] < col_entry[b];
            });

            for(int j = 0; j < row_nnz; j++)
            {
                csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
                csr_val_C[row_begin + j]     = val_entry[perm[j]];
            }
        }
    }

    //-------------------------------------------------------------------------------
    // Compute C = alpha * A + beta * B
    //-------------------------------------------------------------------------------
    static void host_csrgeam_nnz_impl(int        m,
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
                                      int*       nnz_C)
    {
        ROUTINE_TRACE("host_csrgeam_nnz_impl");

        csr_row_ptr_C[0] = 0;

        for(int i = 0; i < m; i++)
        {
            std::vector<int> nnz(n, -1);

            csr_row_ptr_C[i] = 0;

            int row_begin_A = csr_row_ptr_A[i];
            int row_end_A   = csr_row_ptr_A[i + 1];

            for(int j = row_begin_A; j < row_end_A; i++)
            {
                nnz[csr_col_ind_A[j]] = 1;
            }

            int row_begin_B = csr_row_ptr_B[i];
            int row_end_B   = csr_row_ptr_B[i + 1];

            for(int j = row_begin_B; j < row_end_B; j++)
            {
                nnz[csr_col_ind_B[j]] = 1;
            }

            for(int j = 0; j < n; j++)
            {
                if(nnz[j] != -1)
                {
                    csr_row_ptr_C[i + 1]++;
                }
            }
        }

        for(int i = 0; i < m; i++)
        {
            csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
        }

        *nnz_C = csr_row_ptr_C[m];
    }

    static void host_csrgeam_impl(int           m,
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
                                  double*       csr_val_C)
    {
        ROUTINE_TRACE("host_csrgeam_impl");

        for(int i = 0; i < m; i++)
        {
            std::vector<int> nnz(n, -1);

            int row_begin_C = csr_row_ptr_C[i];

            int row_begin_A = csr_row_ptr_A[i];
            int row_end_A   = csr_row_ptr_A[i + 1];

            for(int j = row_begin_A; j < row_end_A; j++)
            {
                csr_col_ind_C[row_begin_C] = csr_col_ind_A[j];
                csr_val_C[row_begin_C]     = alpha * csr_val_A[j];

                nnz[csr_col_ind_A[j]] = row_begin_C;
                row_begin_C++;
            }

            int row_begin_B = csr_row_ptr_B[i];
            int row_end_B   = csr_row_ptr_B[i + 1];

            for(int j = row_begin_B; j < row_end_B; j++)
            {
                int col_B = csr_col_ind_B[j];

                if(nnz[col_B] != -1)
                {
                    csr_val_C[nnz[col_B]] += beta * csr_val_B[j];
                }
                else
                {
                    csr_col_ind_C[row_begin_C] = csr_col_ind_B[j];
                    csr_val_C[row_begin_C]     = beta * csr_val_B[j];

                    nnz[col_B] = row_begin_C;
                    row_begin_C++;
                }
            }
        }

        for(int i = 0; i < m; ++i)
        {
            int row_begin_C = csr_row_ptr_C[i];
            int row_end_C   = csr_row_ptr_C[i + 1];

            int row_nnz = row_end_C - row_begin_C;

            std::vector<int> perm(row_nnz);
            for(int j = 0; j < row_nnz; ++j)
            {
                perm[j] = j;
            }

            std::vector<int>    columns(row_nnz);
            std::vector<double> values(row_nnz);

            for(int j = 0; j < row_nnz; j++)
            {
                columns[j] = csr_col_ind_C[row_begin_C + j];
                values[j]  = csr_val_C[row_begin_C + j];
            }

            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                return columns[a] < columns[b];
            });

            for(int j = 0; j < row_nnz; ++j)
            {
                csr_col_ind_C[row_begin_C + j] = columns[perm[j]];
                csr_val_C[row_begin_C + j]     = values[perm[j]];
            }
        }
    }

    static double get_diagonal_value(
        int col, int diag_index, const double* csr_val, int* structural_zero, int* numeric_zero)
    {
        ROUTINE_TRACE("get_diagonal_value");

        double diag_val = 1.0;
        if(diag_index == -1)
        {
            // Structural zero. No diagonal value exist in matrix. Use diagonal value of 1
            *structural_zero = std::min(*structural_zero, col);
        }
        else
        {
            diag_val = csr_val[diag_index];
            if(diag_val == 0.0)
            {
                // Numerical zero. Use diagonal value of 1 to avoid inf/nan
                *numeric_zero = std::min(*numeric_zero, col);
                diag_val      = 1.0;
            }
        }

        return diag_val;
    }

    //-------------------------------------------------------------------------------
    // Compute incomplete LU factorization inplace
    //-------------------------------------------------------------------------------
    static void host_csrilu0_impl(int        m,
                                  int        n,
                                  int        nnz,
                                  const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  double*    csr_val,
                                  int*       structural_zero,
                                  int*       numeric_zero)
    {
        ROUTINE_TRACE("host_csrilu0_impl");

        std::vector<int> diag_ptr(m, -1);

        for(int row = 0; row < m; row++)
        {
            int row_begin = csr_row_ptr[row];
            int row_end   = csr_row_ptr[row + 1];

            std::vector<int> col_offset_map(n);
            for(int j = 0; j < n; j++)
            {
                col_offset_map[j] = -1;
            }

            for(int j = row_begin; j < row_end; j++)
            {
                col_offset_map[csr_col_ind[j]] = j;
            }

            for(int j = row_begin; j < row_end; j++)
            {
                int col_j = csr_col_ind[j];

                if(col_j < row)
                {
                    int    diag_index = diag_ptr[col_j];
                    double diag_val   = get_diagonal_value(
                        col_j, diag_index, csr_val, structural_zero, numeric_zero);

                    int row_end_col_j = csr_row_ptr[col_j + 1];

                    csr_val[j] = csr_val[j] / diag_val;

                    for(int k = diag_index + 1; k < row_end_col_j; k++)
                    {
                        int col_k = csr_col_ind[k];

                        int col_k_index = col_offset_map[col_k];
                        if(col_k_index != -1)
                        {
                            csr_val[col_k_index] = csr_val[col_k_index] - csr_val[j] * csr_val[k];
                        }
                    }
                }
                else if(col_j == row)
                {
                    diag_ptr[row] = j;
                    break;
                }
                else
                {
                    break;
                }
            }
        }
    }

    //----------------------------------------------------------------------------------------
    // Compute incomplete Cholesky factorization inplace (only modifies lower triangular part)
    //----------------------------------------------------------------------------------------
    static void host_csric0_impl(int        m,
                                 int        n,
                                 int        nnz,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 double*    csr_val,
                                 int*       structural_zero,
                                 int*       numeric_zero)
    {
        ROUTINE_TRACE("host_csric0_impl");

        std::vector<int> diag_ptr(m, -1);

        for(int row = 0; row < m; row++)
        {
            int row_begin = csr_row_ptr[row];
            int row_end   = csr_row_ptr[row + 1];

            std::vector<int> col_offset_map(n);
            for(int j = 0; j < n; j++)
            {
                col_offset_map[j] = -1;
            }

            for(int j = row_begin; j < row_end; j++)
            {
                col_offset_map[csr_col_ind[j]] = j;
            }

            double sum = 0.0;

            for(int j = row_begin; j < row_end; j++)
            {
                int col_j = csr_col_ind[j];

                if(col_j < row)
                {
                    int row_begin_col_j = csr_row_ptr[col_j];
                    int diag_index      = diag_ptr[col_j];

                    double s = 0.0;
                    for(int k = row_begin_col_j; k < diag_index; k++)
                    {
                        int col_k = csr_col_ind[k];

                        int col_k_index = col_offset_map[col_k];
                        if(col_k_index != -1)
                        {
                            s = s + csr_val[col_k_index] * csr_val[k];
                        }
                    }

                    double diag_val = get_diagonal_value(
                        col_j, diag_index, csr_val, structural_zero, numeric_zero);

                    double val = (csr_val[j] - s) / diag_val;

                    sum = sum + val * val;

                    csr_val[j] = val;
                }
                else if(col_j == row)
                {
                    diag_ptr[row] = j;

                    csr_val[j] = std::sqrt(std::abs(csr_val[j] - sum));
                    break;
                }
                else
                {
                    break;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------
    // solve Lx = b where L is a lower triangular sparse matrix
    //-------------------------------------------------------------------------------
    static void host_forward_solve_impl(const int*    csr_row_ptr,
                                        const int*    csr_col_ind,
                                        const double* csr_val,
                                        const double* b,
                                        double*       x,
                                        int           n,
                                        bool          unit_diag)
    {
        ROUTINE_TRACE("host_forward_solve_impl");

        for(int i = 0; i < n; i++)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            assert(row_start >= 0);
            //assert(row_start < nnz);
            assert(row_end >= 0);
            //assert(row_end <= nnz);

            double diag_val = 1.0;

            x[i] = b[i];
            for(int j = row_start; j < row_end; j++)
            {
                int col = csr_col_ind[j];

                assert(col >= 0);
                assert(col < n);

                if(col < i)
                {
                    x[i] -= csr_val[j] * x[col];
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
            x[i] /= diag_val;
        }
    }

    //-------------------------------------------------------------------------------
    // solve Ux = b where U is a upper triangular sparse matrix
    //-------------------------------------------------------------------------------
    static void host_backward_solve_impl(const int*    csr_row_ptr,
                                         const int*    csr_col_ind,
                                         const double* csr_val,
                                         const double* b,
                                         double*       x,
                                         int           n,
                                         bool          unit_diag)
    {
        ROUTINE_TRACE("host_backward_solve_impl");

        for(int i = n - 1; i >= 0; i--)
        {
            int row_start = csr_row_ptr[i];
            int row_end   = csr_row_ptr[i + 1];

            double diag_val = 1.0;

            x[i] = b[i];
            for(int j = row_end - 1; j >= row_start; j--)
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

// Compute y = alpha * x + y
void linalg::host_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_axpy");

    host_axpy_impl(x.get_size(), alpha, x.get_vec(), y.get_vec());
}

// Compute y = alpha * x + beta * y
void linalg::host_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_axpby");

    host_axpby_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec());
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::host_axpbypgz(double                alpha,
                           const vector<double>& x,
                           double                beta,
                           const vector<double>& y,
                           double                gamma,
                           vector<double>&       z)
{
    ROUTINE_TRACE("linalg::host_axpbypgz");

    host_axpbypgz_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec());
}

// Compute y = A * x
void linalg::host_matrix_vector_product(const csr_matrix&     A,
                                        const vector<double>& x,
                                        vector<double>&       y)
{
    ROUTINE_TRACE("linalg::host_matrix_vector_product");

    host_matrix_vector_product_impl(
        A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), y.get_vec(), A.get_m());
}

// Compute y = alpha * A * x + beta * y
void linalg::host_matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_matrix_vector_product");

    host_csrmv_impl(A.get_m(),
                    A.get_n(),
                    A.get_nnz(),
                    alpha,
                    A.get_row_ptr(),
                    A.get_col_ind(),
                    A.get_val(),
                    x.get_vec(),
                    beta,
                    y.get_vec());
}

// Compute C = A * B
void linalg::host_matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::host_matrix_matrix_product");

    // Compute C = A * B
    double alpha = 1.0;
    double beta  = 0.0;

    // Determine number of non-zeros in C = A * B product
    C.resize(A.get_m(), B.get_n(), 0);

    int nnz_C;
    host_csrgemm_nnz_impl(A.get_m(),
                          B.get_n(),
                          A.get_n(),
                          A.get_nnz(),
                          B.get_nnz(),
                          0,
                          alpha,
                          A.get_row_ptr(),
                          A.get_col_ind(),
                          B.get_row_ptr(),
                          B.get_col_ind(),
                          beta,
                          nullptr,
                          nullptr,
                          C.get_row_ptr(),
                          &nnz_C);

    C.resize(A.get_m(), B.get_n(), nnz_C);

    host_csrgemm_impl(A.get_m(),
                      B.get_n(),
                      A.get_n(),
                      A.get_nnz(),
                      B.get_nnz(),
                      0,
                      alpha,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      B.get_row_ptr(),
                      B.get_col_ind(),
                      B.get_val(),
                      beta,
                      nullptr,
                      nullptr,
                      nullptr,
                      C.get_row_ptr(),
                      C.get_col_ind(),
                      C.get_val());
}

// Compute C = A + B
void linalg::host_matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::host_matrix_matrix_addition");

    // Compute C = A + B
    double alpha = 1.0;
    double beta  = 1.0;

    // Determine number of non-zeros in C = A + B product
    C.resize(A.get_m(), B.get_n(), 0);

    int nnz_C;
    host_csrgeam_nnz_impl(A.get_m(),
                          B.get_n(),
                          A.get_nnz(),
                          B.get_nnz(),
                          alpha,
                          A.get_row_ptr(),
                          A.get_col_ind(),
                          beta,
                          B.get_row_ptr(),
                          B.get_col_ind(),
                          C.get_row_ptr(),
                          &nnz_C);

    C.resize(A.get_m(), B.get_n(), nnz_C);

    host_csrgeam_impl(A.get_m(),
                      B.get_n(),
                      A.get_nnz(),
                      B.get_nnz(),
                      alpha,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      beta,
                      B.get_row_ptr(),
                      B.get_col_ind(),
                      B.get_val(),
                      C.get_row_ptr(),
                      C.get_col_ind(),
                      C.get_val());
}

// Incomplete IC factorization
void linalg::host_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::host_csric0");

    host_csric0_impl(LL.get_m(),
                     LL.get_n(),
                     LL.get_nnz(),
                     LL.get_row_ptr(),
                     LL.get_col_ind(),
                     LL.get_val(),
                     structural_zero,
                     numeric_zero);
}

// Incomplete LU factorization
void linalg::host_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::host_csrilu0");

    host_csrilu0_impl(LU.get_m(),
                      LU.get_n(),
                      LU.get_nnz(),
                      LU.get_row_ptr(),
                      LU.get_col_ind(),
                      LU.get_val(),
                      structural_zero,
                      numeric_zero);
}

// Forward solve
void linalg::host_forward_solve(const csr_matrix&     A,
                                const vector<double>& b,
                                vector<double>&       x,
                                bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::host_forward_solve");

    host_forward_solve_impl(A.get_row_ptr(),
                            A.get_col_ind(),
                            A.get_val(),
                            b.get_vec(),
                            x.get_vec(),
                            A.get_m(),
                            unit_diag);
}

// Backward solve
void linalg::host_backward_solve(const csr_matrix&     A,
                                 const vector<double>& b,
                                 vector<double>&       x,
                                 bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::host_backward_solve");

    host_backward_solve_impl(A.get_row_ptr(),
                             A.get_col_ind(),
                             A.get_val(),
                             b.get_vec(),
                             x.get_vec(),
                             A.get_m(),
                             unit_diag);
}

// Transpose matrix
void linalg::host_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::host_transpose_matrix");

    transposeA.resize(A.get_n(), A.get_m(), A.get_nnz());

    int*    csr_row_ptr_T = transposeA.get_row_ptr();
    int*    csr_col_ind_T = transposeA.get_col_ind();
    double* csr_val_T     = transposeA.get_val();

    // Fill arrays
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
        int row_start = csr_row_ptr_A[i];
        int row_end   = csr_row_ptr_A[i + 1];

        for(int j = row_start; j < row_end; j++)
        {
            csr_row_ptr_T[csr_col_ind_A[j] + 1]++;
        }
    }

    // Exclusive scan on row pointer array
    for(int i = 0; i < transposeA.get_m(); i++)
    {
        csr_row_ptr_T[i + 1] += csr_row_ptr_T[i];
    }

    for(int i = 0; i < A.get_m(); i++)
    {
        int row_start = csr_row_ptr_A[i];
        int row_end   = csr_row_ptr_A[i + 1];

        for(int j = row_start; j < row_end; j++)
        {
            int    col = csr_col_ind_A[j];
            double val = csr_val_A[j];

            int start = csr_row_ptr_T[col];
            int end   = csr_row_ptr_T[col + 1];

            for(int k = start; k < end; k++)
            {
                if(csr_col_ind_T[k] == -1)
                {
                    csr_col_ind_T[k] = i;
                    csr_val_T[k]     = val;
                    break;
                }
            }
        }
    }
}

// Dot product
double linalg::host_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_dot_product");

    return host_dot_product_impl(x.get_vec(), y.get_vec(), x.get_size());
}

// Compute residual
void linalg::host_compute_residual(const csr_matrix&     A,
                                   const vector<double>& x,
                                   const vector<double>& b,
                                   vector<double>&       res)
{
    ROUTINE_TRACE("linalg::host_compute_residual");

    host_compute_residual_impl(A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               x.get_vec(),
                               b.get_vec(),
                               res.get_vec(),
                               A.get_m());
}

// Exclusive scan
void linalg::host_exclusive_scan(vector<double>& x)
{
    ROUTINE_TRACE("linalg::host_exclusive_scan");

    host_exclusive_scan_impl(x.get_vec(), x.get_size());
}

// Extract diagonal entries
void linalg::host_diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::host_diagonal");

    host_diagonal_impl(A.get_row_ptr(), A.get_col_ind(), A.get_val(), d.get_vec(), A.get_m());
}

// Euclidean norm
double linalg::host_norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::host_norm_euclid");

    return std::sqrt(host_dot_product(array, array));
}

// Infinity norm
double linalg::host_norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::host_norm_inf");

    return host_norm_inf_impl(array.get_vec(), array.get_size());
}

// Jacobi solve
void linalg::host_jacobi_solve(const vector<double>& rhs,
                               const vector<double>& diag,
                               vector<double>&       x)
{
    ROUTINE_TRACE("linalg::host_jacobi_solve");

    host_jacobi_solve_impl(rhs.get_vec(), diag.get_vec(), x.get_vec(), x.get_size());
}
