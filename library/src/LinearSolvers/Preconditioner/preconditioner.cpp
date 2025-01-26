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
#include "../../../include/LinearSolvers/Preconditioner/preconditioner.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "../../../include/LinearSolvers/AMG/saamg.h"
#include "math.h"
#include <iostream>
#include <vector>

jacobi_precond::jacobi_precond() {}
jacobi_precond::~jacobi_precond() {}

void jacobi_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    diag.resize(m);
    diagonal(csr_row_ptr, csr_col_ind, csr_val, diag.data(), m);
}

void jacobi_precond::solve(const double* rhs, double* x, int n) const
{
    for (int i = 0; i < n; i++)
    {
        x[i] = rhs[i] / diag[i];
    }
}

ilu_precond::ilu_precond() {}
ilu_precond::~ilu_precond() {}

void ilu_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    csr_row_ptr_LU.resize(m + 1);
    csr_col_ind_LU.resize(nnz);
    csr_val_LU.resize(nnz);

    // Copy A to LU
    for(int i = 0; i < m + 1; i++)
    {
        csr_row_ptr_LU[i] = csr_row_ptr[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        csr_col_ind_LU[i] = csr_col_ind[i];
        csr_val_LU[i] = csr_val[i];
    }

    int structural_zero = -1;
    int numeric_zero = -1;

    // In place incomplete LU factorization
    csrilu0(m, n, nnz, csr_row_ptr_LU.data(), csr_col_ind_LU.data(), csr_val_LU.data(), &structural_zero, &numeric_zero);

    //std::cout << "structural_zero: " << structural_zero << " numeric_zero: " << numeric_zero << std::endl;

    //print("LU", csr_row_ptr_LU.data(), csr_col_ind_LU.data(), csr_val_LU.data(), m, n, nnz);
}

void ilu_precond::solve(const double* rhs, double* x, int n) const
{
    // L * U * x = rhs
    // Let y = U * x 
    std::vector<double> y(n);

    // Solve L * y = rhs
    forward_solve(csr_row_ptr_LU.data(), csr_col_ind_LU.data(), csr_val_LU.data(), rhs, y.data(), n, true); 

    // Solve U * x = y
    backward_solve(csr_row_ptr_LU.data(), csr_col_ind_LU.data(), csr_val_LU.data(), y.data(), x, n, false); 
}


ic_precond::ic_precond() {}
ic_precond::~ic_precond() {}

void ic_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    csr_row_ptr_LLT.resize(m + 1);
    csr_col_ind_LLT.resize(nnz);
    csr_val_LLT.resize(nnz);

    // Copy A to LLT
    for(int i = 0; i < m + 1; i++)
    {
        csr_row_ptr_LLT[i] = csr_row_ptr[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        csr_col_ind_LLT[i] = csr_col_ind[i];
        csr_val_LLT[i] = csr_val[i];
    }

    int structural_zero = -1;
    int numeric_zero = -1;

    // In place incomplete Cholesky factorization
    csric0(m, n, nnz, csr_row_ptr_LLT.data(), csr_col_ind_LLT.data(), csr_val_LLT.data(), &structural_zero, &numeric_zero);

    // Fill inplace the upper triangular part with L^T
    for(int row = 0; row < m; row++)
    {
        int start = csr_row_ptr_LLT[row];
        int end = csr_row_ptr_LLT[row + 1];

        for(int j = start; j < end; j++)
        {
            int col = csr_col_ind_LLT[j];

            if(col < row)
            {
                double val = csr_val_LLT[j];

                int start2 = csr_row_ptr_LLT[col];
                int end2 = csr_row_ptr_LLT[col + 1];

                for(int k = start2; k < end2; k++)
                {   
                    if(csr_col_ind_LLT[k] == row)
                    {
                        csr_val_LLT[k] = val;
                        break;
                    }
                }
            }
        }
    }

    //std::cout << "structural_zero: " << structural_zero << " numeric_zero: " << numeric_zero << std::endl;

    //print("LLT", csr_row_ptr_LLT.data(), csr_col_ind_LLT.data(), csr_val_LLT.data(), m, n, nnz);
}

void ic_precond::solve(const double* rhs, double* x, int n) const
{
    // L * L^T * x = rhs
    // Let y = L^T * x 
    std::vector<double> y(n);

    // Solve L * y = rhs
    forward_solve(csr_row_ptr_LLT.data(), csr_col_ind_LLT.data(), csr_val_LLT.data(), rhs, y.data(), n, false); 

    // Solve L^T * x = y
    backward_solve(csr_row_ptr_LLT.data(), csr_col_ind_LLT.data(), csr_val_LLT.data(), y.data(), x, n, false); 
}

saamg_precond::saamg_precond(int presmoothing, int postsmoothing, Cycle cycle, Smoother smoother) 
    : presmoothing(presmoothing), 
      postsmoothing(postsmoothing),
      cycle(cycle),
      smoother(smoother) {}
saamg_precond::~saamg_precond() {}

void saamg_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    saamg_setup(csr_row_ptr, csr_col_ind, csr_val, m, m, nnz, 100, hierachy);
}

void saamg_precond::solve(const double* rhs, double* x, int n) const
{
    amg_solve(hierachy, x, rhs, presmoothing, postsmoothing, 1e-8, cycle, smoother);
}