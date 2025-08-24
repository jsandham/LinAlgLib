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
#include "../../../include/iterative_solvers/preconditioner/preconditioner.h"
#include "../../../include/iterative_solvers/amg/saamg.h"
#include "../../../include/linalg_math.h"

#include <iostream>
#include <vector>

using namespace linalg;

jacobi_precond::jacobi_precond()
    : on_host(false)
{
}
jacobi_precond::~jacobi_precond() {}

void jacobi_precond::build(const csr_matrix& A)
{
    diag.resize(A.get_m());
    diagonal(A, diag);
}

void jacobi_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    // bool moved_to_host = !rhs.is_on_host();
    // if(moved_to_host)
    // {
    //     rhs.move_to_host();
    //     x.move_to_host();
    //     diag.move_to_host();
    // }

    // Solve M * x = rhs where M = D
    jacobi_solve(rhs, diag, x);

    // if(moved_to_host)
    // {
    //     rhs.move_to_device();
    //     x.move_to_device();
    //     diag.move_to_device();
    // }
}

void jacobi_precond::move_to_device()
{
    diag.move_to_device();
    this->on_host = false;
}

void jacobi_precond::move_to_host()
{
    diag.move_to_host();
    this->on_host = true;
}

bool jacobi_precond::is_on_host() const
{
    return on_host;
}

gauss_seidel_precond::gauss_seidel_precond()
    : on_host(false)
{
}
gauss_seidel_precond::~gauss_seidel_precond() {}

void gauss_seidel_precond::build(const csr_matrix& A)
{
    this->A.copy_from(A);
}

void gauss_seidel_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    // Solve M * x = rhs where M = L + D
    forward_solve(this->A, rhs, x, false);
}

void gauss_seidel_precond::move_to_device()
{
    this->on_host = false;
}

void gauss_seidel_precond::move_to_host()
{
    this->on_host = true;
}

bool gauss_seidel_precond::is_on_host() const
{
    return on_host;
}

SOR_precond::SOR_precond(double omega)
    : omega(omega)
    , on_host(false)
{
}
SOR_precond::~SOR_precond() {}

void SOR_precond::build(const csr_matrix& A)
{
    this->A.copy_from(A);

    diag.resize(A.get_m());
    diagonal(A, diag);
}

void SOR_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    const int*    csr_row_ptr = this->A.get_row_ptr();
    const int*    csr_col_ind = this->A.get_col_ind();
    const double* csr_val     = this->A.get_val();

    // Solve M * x = rhs where M = (1 / omega) * (D + omega * L)
    for(int i = 0; i < this->A.get_m(); i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end   = csr_row_ptr[i + 1];

        double diag_val = 1.0;

        x[i] = rhs[i];
        for(int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind[j];
            if(col < i)
            {
                x[i] -= csr_val[j] * x[col];
            }
            else if(col == i)
            {
                diag_val = csr_val[j] / omega;
            }
        }
        x[i] /= diag_val;
    }
}

void SOR_precond::move_to_device()
{
    this->on_host = false;
}

void SOR_precond::move_to_host()
{
    this->on_host = true;
}

bool SOR_precond::is_on_host() const
{
    return on_host;
}

symmetric_gauss_seidel_precond::symmetric_gauss_seidel_precond()
    : on_host(false)
{
}
symmetric_gauss_seidel_precond::~symmetric_gauss_seidel_precond() {}

void symmetric_gauss_seidel_precond::build(const csr_matrix& A)
{
    this->A.copy_from(A);

    diag.resize(A.get_m());
    diagonal(A, diag);
}

void symmetric_gauss_seidel_precond::solve(const vector<double>& rhs, vector<double>& x) const
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

    const int*    csr_row_ptr = this->A.get_row_ptr();
    const int*    csr_col_ind = this->A.get_col_ind();
    const double* csr_val     = this->A.get_val();

    vector<double> y(this->A.get_m());

    // Solve (D - E) * D^-1 * y = rhs
    // (I - E * D^-1) * y = rhs
    for(int i = 0; i < this->A.get_m(); i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end   = csr_row_ptr[i + 1];

        double diag_val = 0.0;

        y[i] = rhs[i];
        for(int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind[j];
            if(col < i)
            {
                y[i] -= (csr_val[j] / diag[col]) * y[col];
            }
            else if(col == i)
            {
                diag_val = (1.0 + csr_val[j] / diag[col]);
            }
        }

        y[i] /= diag_val;
    }

    // Solve (D - F) * x = y
    backward_solve(this->A, y, x, false);
}

void symmetric_gauss_seidel_precond::move_to_device()
{
    this->on_host = false;
}

void symmetric_gauss_seidel_precond::move_to_host()
{
    this->on_host = true;
}

bool symmetric_gauss_seidel_precond::is_on_host() const
{
    return on_host;
}

ilu_precond::ilu_precond()
    : on_host(false)
{
}
ilu_precond::~ilu_precond() {}

void ilu_precond::build(const csr_matrix& A)
{
    this->LU.copy_from(A);

    int structural_zero = -1;
    int numeric_zero    = -1;

    // In place incomplete LU factorization
    csrilu0(LU, &structural_zero, &numeric_zero);
}

void ilu_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    // L * U * x = rhs
    // Let y = U * x
    vector<double> y(rhs.get_size());

    // Solve L * y = rhs
    forward_solve(LU, rhs, y, true);

    // Solve U * x = y
    backward_solve(LU, y, x, false);
}

void ilu_precond::move_to_device()
{
    this->on_host = false;
}

void ilu_precond::move_to_host()
{
    this->on_host = true;
}

bool ilu_precond::is_on_host() const
{
    return on_host;
}

ic_precond::ic_precond()
    : on_host(false)
{
}
ic_precond::~ic_precond() {}

void ic_precond::build(const csr_matrix& A)
{
    this->LLT.copy_from(A);

    int structural_zero = -1;
    int numeric_zero    = -1;

    int     m               = LLT.get_m();
    int     n               = LLT.get_n();
    int     nnz             = LLT.get_nnz();
    int*    csr_row_ptr_LLT = LLT.get_row_ptr();
    int*    csr_col_ind_LLT = LLT.get_col_ind();
    double* csr_val_LLT     = LLT.get_val();

    // In place incomplete Cholesky factorization
    csric0(LLT, &structural_zero, &numeric_zero);

    // Fill inplace the upper triangular part with L^T
    for(int row = 0; row < m; row++)
    {
        int start = csr_row_ptr_LLT[row];
        int end   = csr_row_ptr_LLT[row + 1];

        for(int j = start; j < end; j++)
        {
            int col = csr_col_ind_LLT[j];

            if(col < row)
            {
                double val = csr_val_LLT[j];

                int start2 = csr_row_ptr_LLT[col];
                int end2   = csr_row_ptr_LLT[col + 1];

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
}

void ic_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    // L * L^T * x = rhs
    // Let y = L^T * x
    vector<double> y(rhs.get_size());

    // Solve L * y = rhs
    forward_solve(LLT, rhs, y, false);

    // Solve L^T * x = y
    backward_solve(LLT, y, x, false);
}

void ic_precond::move_to_device()
{
    this->on_host = false;
}

void ic_precond::move_to_host()
{
    this->on_host = true;
}

bool ic_precond::is_on_host() const
{
    return on_host;
}
