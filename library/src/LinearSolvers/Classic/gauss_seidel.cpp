//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include "../../../include/LinearSolvers/Classic/gauss_seidel.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "iostream"

#define DEBUG 1

//-------------------------------------------------------------------------------
// gauss-seidel method
//-------------------------------------------------------------------------------
void gauss_siedel_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                            const double *b, const int n)
{
    for (int j = 0; j < n; j++)
    {
        double sigma = 0.0;
        double ajj = 0.0; // diagonal entry a_jj

        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        for (int k = row_start; k < row_end; k++)
        {
            int col = csr_col_ind[k];
            double val = csr_val[k];
            if (col != j)
            {
                sigma = sigma + val * x[col];
            }
            else
            {
                ajj = val;
            }
        }
        x[j] = (b[j] - sigma) / ajj;
    }
}

int gs(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, const int n,
       const double tol, const int max_iter)
{
    int ii = 0;
    double err = 1.0;
    while (err > tol && ii < max_iter)
    {
        // Gauss-Seidel iteration
        gauss_siedel_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n);

        // err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
        err = fast_error(csr_row_ptr, csr_col_ind, csr_val, x, b, n, tol);

#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif

        ii++;
    }

    return err > tol ? -1 : ii;
}