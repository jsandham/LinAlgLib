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
#include "iostream"
#include "math.h"

//********************************************************************************
//
// Sparse linear algebra functions
//
//********************************************************************************

//-------------------------------------------------------------------------------
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void matrixVectorProduct(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
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
double dotProduct(const double *x, const double *y, const int n)
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
void forwardSolve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
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
void backwardSolve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
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
