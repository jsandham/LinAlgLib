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
    m_diag.resize(m);
    diagonal(csr_row_ptr, csr_col_ind, csr_val, m_diag.data(), m);
}

void jacobi_precond::solve(const double* rhs, double* x, int n) const
{
    // Solve M * x = rhs where M = D
    for (int i = 0; i < n; i++)
    {
        x[i] = rhs[i] / m_diag[i];
    }
}

gauss_seidel_precond::gauss_seidel_precond() {}
gauss_seidel_precond::~gauss_seidel_precond() {}

void gauss_seidel_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    this->m_csr_row_ptr = csr_row_ptr;
    this->m_csr_col_ind = csr_col_ind;
    this->m_csr_val = csr_val;
}

void gauss_seidel_precond::solve(const double* rhs, double* x, int n) const
{
    // Solve M * x = rhs where M = L + D
    forward_solve(m_csr_row_ptr, m_csr_col_ind, m_csr_val, rhs, x, n, false);
}

SOR_precond::SOR_precond(double omega) : m_omega(omega) {}
SOR_precond::~SOR_precond() {}

void SOR_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    this->m_csr_row_ptr = csr_row_ptr;
    this->m_csr_col_ind = csr_col_ind;
    this->m_csr_val = csr_val;

    m_diag.resize(m);
    diagonal(csr_row_ptr, csr_col_ind, csr_val, m_diag.data(), m);
}

void SOR_precond::solve(const double* rhs, double* x, int n) const
{
    // Solve M * x = rhs where M = (1 / omega) * (D + omega * L)
    for (int i = 0; i < n; i++)
    {
        int row_start = m_csr_row_ptr[i];
        int row_end = m_csr_row_ptr[i + 1];

        double diag_val = 1.0;

        x[i] = rhs[i];
        for (int j = row_start; j < row_end; j++)
        {
            int col = m_csr_col_ind[j];
            if (col < i)
            {
                x[i] -= m_csr_val[j] * x[col];
            }
            else if (col == i)
            {
                diag_val = m_csr_val[j] / m_omega;
            }
        }
        x[i] /= diag_val;
    }
}

symmetric_gauss_seidel_precond::symmetric_gauss_seidel_precond() {}
symmetric_gauss_seidel_precond::~symmetric_gauss_seidel_precond() {}

void symmetric_gauss_seidel_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    this->m_csr_row_ptr = csr_row_ptr;
    this->m_csr_col_ind = csr_col_ind;
    this->m_csr_val = csr_val;

    m_diag.resize(m);
    diagonal(csr_row_ptr, csr_col_ind, csr_val, m_diag.data(), m);
}

void symmetric_gauss_seidel_precond::solve(const double* rhs, double* x, int n) const
{
    // M = (D - E) * D^-1 * (D - F) where A = D-E-F
    //
    // |*    |
    // | *-F |
    // |  D  |
    // |-E * |
    // |    *|
    //
    // Solve M * x = rhs
    // (D - E) * D^-1 * (D - F) * x = rhs
    // Let y = (D - F) * x and therefore (D - E) * D^-1 * y = rhs
    // So solve (D - E) * D^-1 * y = rhs followed by (D - F) * x = y

    // Let L = (D - E) * D^-1 and U = (D - F). This gives M = L * U and
    // the error between A and M is A - M = A - L * U = -E * D^-1 * F which 
    // means that if A is diagonally dominant then this error will be small
    // so when A is diagonally dominant, M is a good approximation of A. 

    std::vector<double> y(n);

    // Solve (D - E) * D^-1 * y = rhs
    // (I - E * D^-1) * y = rhs
    for (int i = 0; i < n; i++)
    {
        int row_start = m_csr_row_ptr[i];
        int row_end = m_csr_row_ptr[i + 1];

        y[i] = 1.0;
        for (int j = row_start; j < row_end; j++)
        {
            int col = m_csr_col_ind[j];
            if (col < i)
            {
                y[i] -= (m_csr_val[j] / m_diag[col]) * x[col];
            }
        }
    }

    // Solve (D - F) * x = y
    backward_solve(m_csr_row_ptr, m_csr_col_ind, m_csr_val, y.data(), x, n, false);
}

ilu_precond::ilu_precond() {}
ilu_precond::~ilu_precond() {}

void ilu_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    m_csr_row_ptr_LU.resize(m + 1);
    m_csr_col_ind_LU.resize(nnz);
    m_csr_val_LU.resize(nnz);

    // Copy A to LU
    for(int i = 0; i < m + 1; i++)
    {
        m_csr_row_ptr_LU[i] = csr_row_ptr[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        m_csr_col_ind_LU[i] = csr_col_ind[i];
        m_csr_val_LU[i] = csr_val[i];
    }

    int structural_zero = -1;
    int numeric_zero = -1;

    // In place incomplete LU factorization
    csrilu0(m, n, nnz, m_csr_row_ptr_LU.data(), m_csr_col_ind_LU.data(), m_csr_val_LU.data(), &structural_zero, &numeric_zero);

    //std::cout << "structural_zero: " << structural_zero << " numeric_zero: " << numeric_zero << std::endl;

    //print("LU", csr_row_ptr_LU.data(), csr_col_ind_LU.data(), csr_val_LU.data(), m, n, nnz);
}

void ilu_precond::solve(const double* rhs, double* x, int n) const
{
    // L * U * x = rhs
    // Let y = U * x 
    std::vector<double> y(n);

    // Solve L * y = rhs
    forward_solve(m_csr_row_ptr_LU.data(), m_csr_col_ind_LU.data(), m_csr_val_LU.data(), rhs, y.data(), n, true); 

    // Solve U * x = y
    backward_solve(m_csr_row_ptr_LU.data(), m_csr_col_ind_LU.data(), m_csr_val_LU.data(), y.data(), x, n, false); 
}


ic_precond::ic_precond() {}
ic_precond::~ic_precond() {}

void ic_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    m_csr_row_ptr_LLT.resize(m + 1);
    m_csr_col_ind_LLT.resize(nnz);
    m_csr_val_LLT.resize(nnz);

    // Copy A to LLT
    for(int i = 0; i < m + 1; i++)
    {
        m_csr_row_ptr_LLT[i] = csr_row_ptr[i];
    }

    for(int i = 0; i < nnz; i++)
    {
        m_csr_col_ind_LLT[i] = csr_col_ind[i];
        m_csr_val_LLT[i] = csr_val[i];
    }

    int structural_zero = -1;
    int numeric_zero = -1;

    // In place incomplete Cholesky factorization
    csric0(m, n, nnz, m_csr_row_ptr_LLT.data(), m_csr_col_ind_LLT.data(), m_csr_val_LLT.data(), &structural_zero, &numeric_zero);

    // Fill inplace the upper triangular part with L^T
    for(int row = 0; row < m; row++)
    {
        int start = m_csr_row_ptr_LLT[row];
        int end = m_csr_row_ptr_LLT[row + 1];

        for(int j = start; j < end; j++)
        {
            int col = m_csr_col_ind_LLT[j];

            if(col < row)
            {
                double val = m_csr_val_LLT[j];

                int start2 = m_csr_row_ptr_LLT[col];
                int end2 = m_csr_row_ptr_LLT[col + 1];

                for(int k = start2; k < end2; k++)
                {   
                    if(m_csr_col_ind_LLT[k] == row)
                    {
                        m_csr_val_LLT[k] = val;
                        break;
                    }
                }
            }
        }
    }

    //std::cout << "structural_zero: " << structural_zero << " numeric_zero: " << numeric_zero << std::endl;

    //print("LLT", m_csr_row_ptr_LLT.data(), m_csr_col_ind_LLT.data(), m_csr_val_LLT.data(), m, n, nnz);
}

void ic_precond::solve(const double* rhs, double* x, int n) const
{
    // L * L^T * x = rhs
    // Let y = L^T * x 
    std::vector<double> y(n);

    // Solve L * y = rhs
    forward_solve(m_csr_row_ptr_LLT.data(), m_csr_col_ind_LLT.data(), m_csr_val_LLT.data(), rhs, y.data(), n, false); 

    // Solve L^T * x = y
    backward_solve(m_csr_row_ptr_LLT.data(), m_csr_col_ind_LLT.data(), m_csr_val_LLT.data(), y.data(), x, n, false); 
}

saamg_precond::saamg_precond(int presmoothing, int postsmoothing, Cycle cycle, Smoother smoother) 
    : m_presmoothing(presmoothing), 
      m_postsmoothing(postsmoothing),
      m_cycle(cycle),
      m_smoother(smoother) {}
saamg_precond::~saamg_precond() {}

void saamg_precond::build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz)
{
    saamg_setup(csr_row_ptr, csr_col_ind, csr_val, m, m, nnz, 100, m_hierachy);
}

void saamg_precond::solve(const double* rhs, double* x, int n) const
{
    iter_control control;
    amg_solve(m_hierachy, x, rhs, m_presmoothing, m_postsmoothing, m_cycle, m_smoother, control);
}