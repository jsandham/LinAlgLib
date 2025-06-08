//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019-2025 James Sandham
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

#include "../include/slaf.h"
#include "math.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <vector>

#include "trace.h"

//********************************************************************************
//
// Sparse linear algebra functions
//
//********************************************************************************

//-------------------------------------------------------------------------------
// Compute y = alpha * x + y
//-------------------------------------------------------------------------------
void linalg::axpy(int n, double alpha, const double* x, double* y)
{
    ROUTINE_TRACE("axpy");

    if(alpha == 1.0)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = x[i] + y[i];
        }
    }
    else
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = alpha * x[i] + y[i];
        }
    }
}

void linalg::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    linalg::axpy(x.get_size(), alpha, x.get_vec(), y.get_vec());
}

//-------------------------------------------------------------------------------
// Compute y = alpha * x + beta * y
//-------------------------------------------------------------------------------
void linalg::axpby(int n, double alpha, const double* x, double beta, double* y)
{
    ROUTINE_TRACE("axpby");

    if(alpha == 1.0)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = x[i] + beta * y[i];
        }
    }
    else if (alpha == 0.0)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = beta * y[i];
        }
    }
    else if (beta == 1.0)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = alpha * x[i] + y[i];
        }
    }
    else if (beta == 0.0)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = alpha * x[i];
        }
    }
    else
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for (int i = 0; i < n; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }
}

void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    linalg::axpby(x.get_size(), alpha, x.get_vec(), beta, y.get_vec());
}

//-------------------------------------------------------------------------------
// Compute y = alpha * A * x + beta * y
//-------------------------------------------------------------------------------
void linalg::csrmv(int m, int n, int nnz, double alpha, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val,
           const double *x, double beta, double *y)
{
    ROUTINE_TRACE("csrmv");
    
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (int i = 0; i < m; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double s = 0.0;
        for (int j = row_start; j < row_end; j++)
        {
            s += csr_val[j] * x[csr_col_ind[j]];
        }

        if (beta == 0.0)
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
void linalg::csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                 const int *csr_col_ind_A, const int *csr_row_ptr_B, const int *csr_col_ind_B, double beta,
                 const int *csr_row_ptr_D, const int *csr_col_ind_D, int *csr_row_ptr_C, int *nnz_C)
{
    ROUTINE_TRACE("csrgemm_nnz");
    
    std::vector<int> nnz(n, -1);

    // A is mxk, B is kxn, and C is mxn
    for (int i = 0; i < m + 1; i++)
    {
        csr_row_ptr_C[i] = 0;
    }

    for (int i = 0; i < m; ++i)
    {
        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            int row_begin_B = csr_row_ptr_B[col_A];
            int row_end_B = csr_row_ptr_B[col_A + 1];

            for (int p = row_begin_B; p < row_end_B; p++)
            {
                int col_B = csr_col_ind_B[p];

                if (nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    csr_row_ptr_C[i + 1]++;
                }
            }
        }

        if (beta != 0.0)
        {
            int row_begin_D = csr_row_ptr_D[i];
            int row_end_D = csr_row_ptr_D[i + 1];

            for (int j = row_begin_D; j < row_end_D; j++)
            {
                int col_D = csr_col_ind_D[j];

                if (nnz[col_D] != i)
                {
                    nnz[col_D] = i;
                    csr_row_ptr_C[i + 1]++;
                }
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[m];
}

void linalg::csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, double beta, const int *csr_row_ptr_D, const int *csr_col_ind_D,
             const double *csr_val_D, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C)
{
    ROUTINE_TRACE("csrgemm");

    std::vector<int> nnzs(n, -1);

    for (int i = 0; i < m; i++)
    {
        int row_begin_C = csr_row_ptr_C[i];
        int row_end_C = row_begin_C;

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];
            double val_A = alpha * csr_val_A[j];

            int row_begin_B = csr_row_ptr_B[col_A];
            int row_end_B = csr_row_ptr_B[col_A + 1];

            for (int p = row_begin_B; p < row_end_B; p++)
            {
                int col_B = csr_col_ind_B[p];
                double val_B = csr_val_B[p];

                if (nnzs[col_B] < row_begin_C)
                {
                    nnzs[col_B] = row_end_C;
                    csr_col_ind_C[row_end_C] = col_B;
                    csr_val_C[row_end_C] = val_A * val_B;
                    row_end_C++;
                }
                else
                {
                    csr_val_C[nnzs[col_B]] += val_A * val_B;
                }
            }
        }

        if (beta != 0.0)
        {
            int row_begin_D = csr_row_ptr_D[i];
            int row_end_D = csr_row_ptr_D[i + 1];

            for (int j = row_begin_D; j < row_end_D; j++)
            {
                int col_D = csr_col_ind_D[j];
                double val_D = beta * csr_val_D[j];

                // Check if a new nnz is generated or if the value is added
                if (nnzs[col_D] < row_begin_C)
                {
                    nnzs[col_D] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_D;
                    csr_val_C[row_end_C] = val_D;
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

    std::vector<int> cols(nnz);
    std::vector<double> vals(nnz);

    memcpy(cols.data(), csr_col_ind_C, sizeof(int) * nnz);
    memcpy(vals.data(), csr_val_C, sizeof(double) * nnz);

    for (int i = 0; i < m; i++)
    {
        int row_begin = csr_row_ptr_C[i];
        int row_end = csr_row_ptr_C[i + 1];
        int row_nnz = row_end - row_begin;

        std::vector<int> perm(row_nnz);
        for (int j = 0; j < row_nnz; j++)
        {
            perm[j] = j;
        }

        int *col_entry = cols.data() + row_begin;
        double *val_entry = vals.data() + row_begin;

        std::sort(perm.begin(), perm.end(), [&](const int &a, const int &b) { return col_entry[a] < col_entry[b]; });

        for (int j = 0; j < row_nnz; j++)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j] = val_entry[perm[j]];
        }
    }
}

//-------------------------------------------------------------------------------
// Compute C = alpha * A + beta * B
//-------------------------------------------------------------------------------
void linalg::csrgeam_nnz(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
                 double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B, int *csr_row_ptr_C, int *nnz_C)
{
    ROUTINE_TRACE("csrgeam_nnz");

    csr_row_ptr_C[0] = 0;

    for (int i = 0; i < m; i++)
    {
        std::vector<int> nnz(n, -1);

        csr_row_ptr_C[i] = 0;

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; i++)
        {
            nnz[csr_col_ind_A[j]] = 1;
        }

        int row_begin_B = csr_row_ptr_B[i];
        int row_end_B = csr_row_ptr_B[i + 1];

        for (int j = row_begin_B; j < row_end_B; j++)
        {
            nnz[csr_col_ind_B[j]] = 1;
        }

        for (int j = 0; j < n; j++)
        {
            if (nnz[j] != -1)
            {
                csr_row_ptr_C[i + 1]++;
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[m];
}

void linalg::csrgeam(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
             const double *csr_val_A, double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C)
{
    ROUTINE_TRACE("csrgeam");

    for (int i = 0; i < m; i++)
    {
        std::vector<int> nnz(n, -1);

        int row_begin_C = csr_row_ptr_C[i];

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; j++)
        {
            csr_col_ind_C[row_begin_C] = csr_col_ind_A[j];
            csr_val_C[row_begin_C] = alpha * csr_val_A[j];

            nnz[csr_col_ind_A[j]] = row_begin_C;
            row_begin_C++;
        }

        int row_begin_B = csr_row_ptr_B[i];
        int row_end_B = csr_row_ptr_B[i + 1];

        for (int j = row_begin_B; j < row_end_B; j++)
        {
            int col_B = csr_col_ind_B[j];

            if (nnz[col_B] != -1)
            {
                csr_val_C[nnz[col_B]] += beta * csr_val_B[j];
            }
            else
            {
                csr_col_ind_C[row_begin_C] = csr_col_ind_B[j];
                csr_val_C[row_begin_C] = beta * csr_val_B[j];

                nnz[col_B] = row_begin_C;
                row_begin_C++;
            }
        }
    }

    for (int i = 0; i < m; ++i)
    {
        int row_begin_C = csr_row_ptr_C[i];
        int row_end_C = csr_row_ptr_C[i + 1];

        int row_nnz = row_end_C - row_begin_C;

        std::vector<int> perm(row_nnz);
        for (int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        std::vector<int> columns(row_nnz);
        std::vector<double> values(row_nnz);

        for (int j = 0; j < row_nnz; j++)
        {
            columns[j] = csr_col_ind_C[row_begin_C + j];
            values[j] = csr_val_C[row_begin_C + j];
        }

        std::sort(perm.begin(), perm.end(), [&](const int &a, const int &b) { return columns[a] < columns[b]; });

        for (int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin_C + j] = columns[perm[j]];
            csr_val_C[row_begin_C + j] = values[perm[j]];
        }
    }
}

namespace linalg
{
static double get_diagonal_value(int col, int diag_index, const double *csr_val, int *structural_zero,
                                 int *numeric_zero)
{
    ROUTINE_TRACE("get_diagonal_value");

    double diag_val = 1.0;
    if (diag_index == -1)
    {
        // Structural zero. No diagonal value exist in matrix. Use diagonal value of 1
        *structural_zero = std::min(*structural_zero, col);
    }
    else
    {
        diag_val = csr_val[diag_index];
        if (diag_val == 0.0)
        {
            // Numerical zero. Use diagonal value of 1 to avoid inf/nan
            *numeric_zero = std::min(*numeric_zero, col);
            diag_val = 1.0;
        }
    }

    return diag_val;
}
}

//-------------------------------------------------------------------------------
// Compute incomplete LU factorization inplace
//-------------------------------------------------------------------------------
void linalg::csrilu0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
             int *structural_zero, int *numeric_zero)
{
    ROUTINE_TRACE("csrilu0");

    std::vector<int> diag_ptr(m, -1);

    for (int row = 0; row < m; row++)
    {
        int row_begin = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];

        std::vector<int> col_offset_map(n);
        for (int j = 0; j < n; j++)
        {
            col_offset_map[j] = -1;
        }

        for (int j = row_begin; j < row_end; j++)
        {
            col_offset_map[csr_col_ind[j]] = j;
        }

        for (int j = row_begin; j < row_end; j++)
        {
            int col_j = csr_col_ind[j];

            if (col_j < row)
            {
                int diag_index = diag_ptr[col_j];
                double diag_val = get_diagonal_value(col_j, diag_index, csr_val, structural_zero, numeric_zero);

                int row_end_col_j = csr_row_ptr[col_j + 1];

                csr_val[j] = csr_val[j] / diag_val;

                for (int k = diag_index + 1; k < row_end_col_j; k++)
                {
                    int col_k = csr_col_ind[k];

                    int col_k_index = col_offset_map[col_k];
                    if (col_k_index != -1)
                    {
                        csr_val[col_k_index] = csr_val[col_k_index] - csr_val[j] * csr_val[k];
                    }
                }
            }
            else if (col_j == row)
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
void linalg::csric0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
            int *structural_zero, int *numeric_zero)
{
    ROUTINE_TRACE("csric0");

    std::vector<int> diag_ptr(m, -1);

    for (int row = 0; row < m; row++)
    {
        int row_begin = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];

        std::vector<int> col_offset_map(n);
        for (int j = 0; j < n; j++)
        {
            col_offset_map[j] = -1;
        }

        for (int j = row_begin; j < row_end; j++)
        {
            col_offset_map[csr_col_ind[j]] = j;
        }

        double sum = 0.0;

        for (int j = row_begin; j < row_end; j++)
        {
            int col_j = csr_col_ind[j];

            if (col_j < row)
            {
                int row_begin_col_j = csr_row_ptr[col_j];
                int diag_index = diag_ptr[col_j];

                double s = 0.0;
                for (int k = row_begin_col_j; k < diag_index; k++)
                {
                    int col_k = csr_col_ind[k];

                    int col_k_index = col_offset_map[col_k];
                    if (col_k_index != -1)
                    {
                        s = s + csr_val[col_k_index] * csr_val[k];
                    }
                }

                double diag_val = get_diagonal_value(col_j, diag_index, csr_val, structural_zero, numeric_zero);

                double val = (csr_val[j] - s) / diag_val;

                sum = sum + val * val;

                csr_val[j] = val;
            }
            else if (col_j == row)
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
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void linalg::matrix_vector_product(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                           double *y, int n)
{
    ROUTINE_TRACE("matrix_vector_product");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double s = 0.0;
        for (int j = row_start; j < row_end; j++)
        {
            s += csr_val[j] * x[csr_col_ind[j]];
        }

        y[i] = s;
    }
}

// Compute C = A * B
// void matrix_matrix_product(int m, int k, int n, int nnz_A, const int *csr_row_ptr_A, const int *csr_col_ind_A, const double *csr_val_A, 
//                                          int nnz_B, const int *csr_row_ptr_B, const int *csr_col_ind_B, const double *csr_val_B,
//                                          int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C)
// {
//     ROUTINE_TRACE("matrix_matrix_product");

//     // Compute C = A * B
//     double alpha = 1.0;
//     double beta = 0.0;

//     // Determine number of non-zeros in C = A * B product
//     C.m = A.m;
//     C.n = B.n;
//     C.nnz = 0;
//     C.hcsr_row_ptr.resize(C.m + 1, 0);

//     csrgemm_nnz(A.m, B.n, A.n, A.nnz, B.nnz, 0, alpha, A.hcsr_row_ptr.data(), A.hcsr_col_ind.data(), B.hcsr_row_ptr.data(),
//                 B.hcsr_col_ind.data(), beta, nullptr, nullptr, C.hcsr_row_ptr.data(), &C.nnz);

//     C.hcsr_col_ind.resize(C.nnz);
//     C.hcsr_val.resize(C.nnz);

//     csrgemm(A.m, B.n, A.n, A.nnz, B.nnz, 0, alpha, A.hcsr_row_ptr.data(), A.hcsr_col_ind.data(), A.hcsr_val.data(),
//             B.hcsr_row_ptr.data(), B.hcsr_col_ind.data(), B.hcsr_val.data(), beta, nullptr, nullptr, nullptr,
//             C.hcsr_row_ptr.data(), C.hcsr_col_ind.data(), C.hcsr_val.data());
// }

//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
double linalg::dot_product(const double *x, const double *y, int n)
{
    ROUTINE_TRACE("dot_product");

    double dot_prod = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+: dot_prod)
#endif
    for (int i = 0; i < n; i++)
    {
        dot_prod += x[i] * y[i];
    }

    return dot_prod;
}

double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("dot_product");

    assert(x.get_size() == y.get_size());

    double dot_prod = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+: dot_prod)
#endif
    for (int i = 0; i < x.get_size(); i++)
    {
        dot_prod += x[i] * y[i];
    }

    return dot_prod;
}


//-------------------------------------------------------------------------------
// fill array with zeros
//-------------------------------------------------------------------------------
void linalg::fill_with_zeros(uint32_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_zeros");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 0;
    }
}
void linalg::fill_with_zeros(int32_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_zeros");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 0;
    }
}
void linalg::fill_with_zeros(int64_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_zeros");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 0;
    }
}
void linalg::fill_with_zeros(double *x, size_t n)
{
    ROUTINE_TRACE("fill_with_zeros");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }
}

//-------------------------------------------------------------------------------
// fill array with ones
//-------------------------------------------------------------------------------
void linalg::fill_with_ones(uint32_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_ones");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1;
    }
}
void linalg::fill_with_ones(int32_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_ones");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1;
    }
}
void linalg::fill_with_ones(int64_t *x, size_t n)
{
    ROUTINE_TRACE("fill_with_ones");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1;
    }
}
void linalg::fill_with_ones(double *x, size_t n)
{
    ROUTINE_TRACE("fill_with_ones");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        x[i] = 1.0;
    }
}

//-------------------------------------------------------------------------------
// exclusive scan
//-------------------------------------------------------------------------------
void linalg::compute_exclusize_scan(double *x, int n)
{
    if(n > 0)
    {
        x[0] = 0;

        for(int i = 0; i < n - 1; i++)
        {
            x[i + 1] += x[i];
        }
    }
}

void linalg::exclusize_scan(vector<double>& x)
{
    if(x.get_size() > 0)
    {
        x[0] = 0;

        for(int i = 0; i < x.get_size() - 1; i++)
        {
            x[i + 1] += x[i];
        }
    }
}

//-------------------------------------------------------------------------------
// Compute residual res = b - A * x
//-------------------------------------------------------------------------------
void linalg::compute_residual(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
    const double* b, double* res, int n)
{
    ROUTINE_TRACE("compute_residual");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double s = 0.0;
        for (int j = row_start; j < row_end; j++)
        {
            s += csr_val[j] * x[csr_col_ind[j]];
        }

        res[i] = b[i] - s;
    }
}

void linalg::compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res)
{
    linalg::compute_residual(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), res.get_vec(), A.get_m());
}

//-------------------------------------------------------------------------------
// copy array
//-------------------------------------------------------------------------------
void linalg::copy(uint32_t* dest, const uint32_t* src, size_t n)
{
    ROUTINE_TRACE("copy");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        dest[i] = src[i];
    }
}
void linalg::copy(int32_t* dest, const int32_t* src, size_t n)
{
    ROUTINE_TRACE("copy");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        dest[i] = src[i];
    }
}
void linalg::copy(int64_t* dest, const int64_t* src, size_t n)
{
    ROUTINE_TRACE("copy");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        dest[i] = src[i];
    }
}
void linalg::copy(double* dest, const double* src, size_t n)
{
    ROUTINE_TRACE("copy");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < n; i++)
    {
        dest[i] = src[i];
    }
}

//-------------------------------------------------------------------------------
// diagonal d = diag(A)
//-------------------------------------------------------------------------------
void linalg::diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, int n)
{
    ROUTINE_TRACE("diagonal");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (csr_col_ind[j] == i)
            {
                d[i] = csr_val[j];
                break;
            }
        }
    }
}

void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    linalg::diagonal(A.get_row_ptr(), A.get_col_ind(), A.get_val(), d.get_vec(), A.get_m());
}

//-------------------------------------------------------------------------------
// solve Lx = b where L is a lower triangular sparse matrix
//-------------------------------------------------------------------------------
void linalg::forward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   int n, bool unit_diag)
{
    ROUTINE_TRACE("forward_solve");

    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        assert(row_start >= 0);
        //assert(row_start < nnz);
        assert(row_end >= 0);
        //assert(row_end <= nnz);

        double diag_val = 1.0;

        x[i] = b[i];
        for (int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind[j];

            assert(col >= 0);
            assert(col < n);

            if (col < i)
            {
                x[i] -= csr_val[j] * x[col];
            }
            else if (!unit_diag && col == i)
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
void linalg::backward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                    int n, bool unit_diag)
{
    ROUTINE_TRACE("backward_solve");

    for (int i = n - 1; i >= 0; i--)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double diag_val = 1.0;

        x[i] = b[i];
        for (int j = row_end - 1; j >= row_start; j--)
        {
            int col = csr_col_ind[j];
            if (col > i)
            {
                x[i] -= csr_val[j] * x[col];
            }
            else if (!unit_diag && col == i)
            {
                diag_val = csr_val[j];
            }
        }

        x[i] /= diag_val;
    }
}

//-------------------------------------------------------------------------------
// error e = |b-A*x|
//-------------------------------------------------------------------------------
double linalg::error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             int n)
{
    ROUTINE_TRACE("error");

    double e = 0.0;
    for (int j = 0; j < n; j++)
    {
        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        double s = 0.0;
        for (int i = row_start; i < row_end; i++)
        {
            s += csr_val[i] * x[csr_col_ind[i]];
        }
        e = e + (b[j] - s) * (b[j] - s);
    }

    return std::sqrt(e);
}

//-------------------------------------------------------------------------------
// error e = |b-A*x| stops calculating error if error goes above tolerance
//-------------------------------------------------------------------------------
double linalg::fast_error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                  const double *b, int n, double tol)
{
    ROUTINE_TRACE("fast_error");
    int j = 0;
    double e = 0.0;
    while (e < tol && j < n)
    {
        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        double s = 0.0;
        for (int i = row_start; i < row_end; i++)
        {
            s += csr_val[i] * x[csr_col_ind[i]];
        }

        e = e + (b[j] - s) * (b[j] - s);
        j++;
    }

    return std::sqrt(e);
}

//-------------------------------------------------------------------------------
// infinity norm
//-------------------------------------------------------------------------------
double linalg::norm_inf(const double *array, int n)
{
    ROUTINE_TRACE("norm_inf");

    double norm = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(max: norm)
#endif
    for (int i = 0; i < n; i++)
    {
        norm = std::max(std::abs(array[i]), norm);
    }

    return norm;
}

double linalg::norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("norm_inf");

    double norm = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(max: norm)
#endif
    for (int i = 0; i < array.get_size(); i++)
    {
        norm = std::max(std::abs(array[i]), norm);
    }

    return norm;
}

//-------------------------------------------------------------------------------
// euclidean norm
//-------------------------------------------------------------------------------
double linalg::norm_euclid(const double *array, int n)
{
    ROUTINE_TRACE("norm_euclid");
    return std::sqrt(dot_product(array, array, n));
}

double linalg::norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("norm_euclid");
    return std::sqrt(dot_product(array, array));
}

//-------------------------------------------------------------------------------
// print matrix to console
//-------------------------------------------------------------------------------
void linalg::print_matrix(const std::string name, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n,
           int nnz)
{
    ROUTINE_TRACE("print_matrix");

    std::cout << name << std::endl;
    for (int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        std::vector<double> temp(n, 0.0);
        for (int j = start; j < end; j++)
        {
            temp[csr_col_ind[j]] = (csr_val != nullptr) ? csr_val[j] : 1.0;
        }

        for (int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;
}
