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

}

void csrgeam(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, double beta, const int *csr_row_ptr_B,
             const int *csr_col_ind_B, const double *csr_val_B, const int *csr_row_ptr_C, int *csr_col_ind_C,
             double *csr_val_C)
{

}




//-------------------------------------------------------------------------------
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void matrix_vector_product(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                         double *y, const int n)
{
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
double dot_product(const double *x, const double *y, const int n)
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
void diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, const int n)
{
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
                  const int n)
{
    for (int i = 0; i < n; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        x[i] = b[i];
        for (int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind[j];
            if (col < i)
            {
                x[i] -= csr_val[j] * x[col];
            }
        }
        x[i] /= csr_val[row_end - 1];
    }
}

//-------------------------------------------------------------------------------
// solve Ux = b where U is a upper triangular sparse matrix
//-------------------------------------------------------------------------------
void backward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   const int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        x[i] = b[i];
        for (int j = row_end - 1; j >= row_start; j--)
        {
            int col = csr_col_ind[j];
            if (col > i)
            {
                x[i] -= csr_val[j] * x[col];
            }
        }

        x[i] /= csr_val[row_start];
    }
}

//-------------------------------------------------------------------------------
// error e = |b-A*x|
//-------------------------------------------------------------------------------
double error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             const int n)
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
                  const double *b, const int n, const double tol)
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
