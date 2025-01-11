//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
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

#include "../../include/LinearSolvers/slaf.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <assert.h>
#include "math.h"

//********************************************************************************
//
// Sparse linear algebra functions
//
//********************************************************************************

//-------------------------------------------------------------------------------
// Compute y = alpha * A * x + beta * y
//-------------------------------------------------------------------------------
void csrmv(int m, int n, int nnz, double alpha, const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, 
           const double* x, double beta, double* y)
{
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
void csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                 const int *csr_col_ind_A, const int *csr_row_ptr_B, const int *csr_col_ind_B, double beta,
                 const int *csr_row_ptr_D, const int *csr_col_ind_D, int *csr_row_ptr_C, int *nnz_C)
{
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

void csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, const int *csr_row_ptr_B,
             const int *csr_col_ind_B, const double *csr_val_B, double beta, const int *csr_row_ptr_D,
             const int *csr_col_ind_D, const double *csr_val_D, const int *csr_row_ptr_C, int *csr_col_ind_C,
             double *csr_val_C)
{
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
void csrgeam_nnz(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A,
                 const int *csr_col_ind_A, double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B,
                 int *csr_row_ptr_C, int *nnz_C)
{
    csr_row_ptr_C[0] = 0;

    for(int i = 0; i < m; i++)
    {
        std::vector<int> nnz(n, -1);

        csr_row_ptr_C[i] = 0;

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for(int j = row_begin_A; j < row_end_A; i++)
        {
            nnz[csr_col_ind_A[j]] = 1;
        }

        int row_begin_B = csr_row_ptr_B[i];
        int row_end_B = csr_row_ptr_B[i + 1];

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

void csrgeam(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, double beta, const int *csr_row_ptr_B,
             const int *csr_col_ind_B, const double *csr_val_B, const int *csr_row_ptr_C, int *csr_col_ind_C,
             double *csr_val_C)
{
    for(int i = 0; i < m; i++)
    {
        std::vector<int> nnz(n, -1);

        int row_begin_C = csr_row_ptr_C[i];

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for(int j = row_begin_A; j < row_end_A; j++)
        {
            csr_col_ind_C[row_begin_C] = csr_col_ind_A[j];
            csr_val_C[row_begin_C] = alpha * csr_val_A[j];

            nnz[csr_col_ind_A[j]] = row_begin_C;
            row_begin_C++;
        }

        int row_begin_B = csr_row_ptr_B[i];
        int row_end_B = csr_row_ptr_B[i + 1];

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
                csr_val_C[row_begin_C] = beta * csr_val_B[j];

                nnz[col_B] = row_begin_C;
                row_begin_C++;
            }
        }
    }

    for(int i = 0; i < m; ++i)
    {
        int row_begin_C = csr_row_ptr_C[i];
        int row_end_C = csr_row_ptr_C[i + 1];

        int row_nnz = row_end_C - row_begin_C;

        std::vector<int> perm(row_nnz);
        for(int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        std::vector<int> columns(row_nnz);
        std::vector<double> values(row_nnz);

        for(int j = 0; j < row_nnz; j++)
        {
            columns[j] = csr_col_ind_C[row_begin_C + j];
            values[j] = csr_val_C[row_begin_C + j];
        }

        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {return columns[a] < columns[b];});

        for(int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin_C + j] = columns[perm[j]];
            csr_val_C[row_begin_C + j]     = values[perm[j]];
        }
    }
}

static double get_diagonal_value(int col, int diag_index, const double* csr_val, int* structural_zero, int* numeric_zero)
{
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
            diag_val = 1.0;
        }
    }

    return diag_val;
}

//-------------------------------------------------------------------------------
// Compute incomplete LU factorization inplace
//-------------------------------------------------------------------------------
void csrilu0(int m, int n, int nnz, const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int* structural_zero, int* numeric_zero)
{
    std::vector<int> diag_ptr(m, -1);

    for(int row = 0; row < m; row++)
    {
        int row_begin = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];

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
                int diag_index = diag_ptr[col_j];
                double diag_val = get_diagonal_value(col_j, diag_index, csr_val, structural_zero, numeric_zero);

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
void csric0(int m, int n, int nnz, const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int* structural_zero, int* numeric_zero)
{
    std::vector<int> diag_ptr(m, -1);

    for(int row = 0; row < m; row++)
    {
        int row_begin = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];

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
                int diag_index = diag_ptr[col_j];

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

                double diag_val = get_diagonal_value(col_j, diag_index, csr_val, structural_zero, numeric_zero);

                double val = (csr_val[j] - s) / diag_val;

                sum = sum + val * val;

                csr_val[j] = val;

                std::cout << "row: " << row << " col_j: " << col_j << " val: " << val << " sum: " << sum << " s: " << s << " diag_val: " << diag_val << std::endl;
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
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void matrix_vector_product(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                         double *y, int n)
{
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

//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
double dot_product(const double *x, const double *y, int n)
{
    double dot_prod = 0.0;
    for (int i = 0; i < n; i++)
    {
        dot_prod = dot_prod + x[i] * y[i];
    }

    return dot_prod;
}

//-------------------------------------------------------------------------------
// diagonal d = diag(A)
//-------------------------------------------------------------------------------
void diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, int n)
{
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

//-------------------------------------------------------------------------------
// solve Lx = b where L is a lower triangular sparse matrix
//-------------------------------------------------------------------------------
void forward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                  int n, bool unit_diag)
{
    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double diag_val = 1.0;

        x[i] = b[i];
        for (int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind[j];
            if (col < i)
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

//-------------------------------------------------------------------------------
// solve Ux = b where U is a upper triangular sparse matrix
//-------------------------------------------------------------------------------
void backward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   int n, bool unit_diag)
{
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
            else if(!unit_diag && col == i)
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
double error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             int n)
{
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

    return sqrt(e);
}

//-------------------------------------------------------------------------------
// error e = |b-A*x| stops calculating error if error goes above tolerance
//-------------------------------------------------------------------------------
double fast_error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                  const double *b, int n, double tol)
{
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

    return sqrt(e);
}

//-------------------------------------------------------------------------------
// infinity norm
//-------------------------------------------------------------------------------
double norm_inf(const double* array, int n)
{
    double norm = 0.0;
    for(int i = 0; i < n; i++)
    {
        norm = std::max(std::abs(array[i]), norm);
    }

    return norm;
}

//-------------------------------------------------------------------------------
// print matrix to console
//-------------------------------------------------------------------------------
void print(const std::string name, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    std::cout << name << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        std::vector<double> temp(n, 0.0);
        for(int j = start; j < end; j++)
        {
            temp[csr_col_ind[j]] = csr_val[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;
}
