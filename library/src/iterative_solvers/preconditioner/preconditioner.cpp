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

#include "../../backend/device/device_math.h"
#include "../../backend/host/host_math.h"

#include "../../trace.h"
#include "../../utility.h"

#include <iostream>
#include <vector>

using namespace linalg;

jacobi_precond::jacobi_precond()
    : on_host(true)
{
}
jacobi_precond::~jacobi_precond() {}

void jacobi_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("jacobi_precond::build");

    diag.resize(A.get_m());
    diagonal(A, diag);
}

void jacobi_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("jacobi_precond::solve");

    // Solve M * x = rhs where M = D
    backend_dispatch("linalg::jacobi_solve", host_jacobi_solve, device_jacobi_solve, rhs, diag, x);
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
    : on_host(true)
{
    create_csrtrsv_descr(&descr_M);
}
gauss_seidel_precond::~gauss_seidel_precond()
{
    destroy_csrtrsv_descr(descr_M);
}

void gauss_seidel_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("gauss_seidel_precond::build");

    this->M.copy_lower_triangular_from(A, false);
    csrtrsv_analysis(this->M, triangular_type::lower, diagonal_type::non_unit, this->descr_M);
}

void gauss_seidel_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("gauss_seidel_precond::solve");

    // Solve M * x = rhs where M = L + D
    csrtrsv_solve(
        this->M, rhs, x, 1.0, triangular_type::lower, diagonal_type::non_unit, this->descr_M);
}

void gauss_seidel_precond::move_to_device()
{
    this->M.move_to_device();
    this->on_host = false;
}

void gauss_seidel_precond::move_to_host()
{
    this->M.move_to_host();
    this->on_host = true;
}

bool gauss_seidel_precond::is_on_host() const
{
    return on_host;
}

SOR_precond::SOR_precond(double omega)
    : omega(omega)
    , on_host(true)
{
    std::cout << "Create SOR preconditioner with omega: " << omega << std::endl;
    create_csrtrsv_descr(&descr_M);
}
SOR_precond::~SOR_precond()
{
    std::cout << "Destroy SOR preconditioner with omega: " << omega << std::endl;
    destroy_csrtrsv_descr(descr_M);
}

void SOR_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("SOR_precond::build");

    // M = (1 / omega) * (D + omega * L)
    //   = (1 / omega) * D + L
    this->M.copy_lower_triangular_from(A, false);
    this->M.scale_diagonal_by(1.0 / omega);

    csrtrsv_analysis(this->M, triangular_type::lower, diagonal_type::non_unit, this->descr_M);
}

void SOR_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("SOR_precond::solve");

    // Solve M * x = rhs where M = (1 / omega) * (D + omega * L)
    csrtrsv_solve(
        this->M, rhs, x, 1.0, triangular_type::lower, diagonal_type::non_unit, this->descr_M);
}

void SOR_precond::move_to_device()
{
    this->M.move_to_device();
    this->on_host = false;
}

void SOR_precond::move_to_host()
{
    this->M.move_to_host();
    this->on_host = true;
}

bool SOR_precond::is_on_host() const
{
    return on_host;
}

symmetric_gauss_seidel_precond::symmetric_gauss_seidel_precond()
    : on_host(true)
{
    create_csrtrsv_descr(&descr_L);
    create_csrtrsv_descr(&descr_U);
}
symmetric_gauss_seidel_precond::~symmetric_gauss_seidel_precond()
{
    destroy_csrtrsv_descr(descr_L);
    destroy_csrtrsv_descr(descr_U);
}

void symmetric_gauss_seidel_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("symmetric_gauss_seidel_precond::build");

    // M = (D - E) * D^-1 * (D - F) where A = D-E-F
    //
    // |*    |
    // | *-F |
    // |  D  |
    // |-E * |
    // |    *|
    //
    // L = (D - E) * D^-1 and U = (D - F) so that M = L * U
    this->L.copy_lower_triangular_from(A, false);
    this->L.scale_by_inverse_diagonal();

    this->U.copy_upper_triangular_from(A, false);

    csrtrsv_analysis(this->L, triangular_type::lower, diagonal_type::non_unit, this->descr_L);
    csrtrsv_analysis(this->U, triangular_type::upper, diagonal_type::non_unit, this->descr_U);
}

void symmetric_gauss_seidel_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("symmetric_gauss_seidel_precond::solve");

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

    vector<double> y(this->L.get_m());

    if(rhs.is_on_host())
    {
        y.move_to_host();
    }
    else
    {
        y.move_to_device();
    }

    // Solve (D - E) * D^-1 * y = L * y = rhs
    csrtrsv_solve(
        this->L, rhs, y, 1.0, triangular_type::lower, diagonal_type::non_unit, this->descr_L);

    // Solve (D - F) * x = y
    csrtrsv_solve(
        this->U, y, x, 1.0, triangular_type::upper, diagonal_type::non_unit, this->descr_U);
}

void symmetric_gauss_seidel_precond::move_to_device()
{
    this->L.move_to_device();
    this->U.move_to_device();
    this->on_host = false;
}

void symmetric_gauss_seidel_precond::move_to_host()
{
    this->L.move_to_host();
    this->U.move_to_host();
    this->on_host = true;
}

bool symmetric_gauss_seidel_precond::is_on_host() const
{
    return on_host;
}

SSOR_precond::SSOR_precond(double omega)
    : omega(omega)
    , on_host(true)
{
    create_csrtrsv_descr(&descr_L);
    create_csrtrsv_descr(&descr_U);
}
SSOR_precond::~SSOR_precond()
{
    destroy_csrtrsv_descr(descr_L);
    destroy_csrtrsv_descr(descr_U);
}

void SSOR_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("SSOR_precond::build");

    // Let alpha = 1 / (omega * (2-omega)) and beta = 1 / omega
    double beta = 1.0 / omega;

    // M = alpha * (beta * D - E) * D^-1 * (beta * D - F), where A = D-E-F
    //
    // |*    |
    // | *-F |
    // |  D  |
    // |-E * |
    // |    *|
    //
    // L = (beta * D - E) * D^-1 and U = (beta * D - F) so that M = L * U
    int m = A.get_m();
    int n = A.get_n();

    // Fill L = (beta * D - E) * D^-1
    L.resize(m, n, 0);

    // Determine non-zero count in lower triangular portion of A
    int nnz_L = 0;
    backend_dispatch("linalg::csr_matrix::copy_lower_triangular_from",
                     host_extract_lower_triangular_nnz,
                     device_extract_lower_triangular_nnz,
                     A,
                     L,
                     nnz_L);

    std::cout << "nnz_L: " << nnz_L << std::endl;

    L.resize(m, n, nnz_L);

    backend_dispatch("linalg::ssor_fill_lower_precond",
                     host_ssor_fill_lower_precond,
                     device_ssor_fill_lower_precond,
                     A,
                     L,
                     beta);

    // Fill U = (beta * D - F)
    U.resize(m, n, 0);

    // Determine non-zero count in upper triangular portion of A
    int nnz_U = 0;

    backend_dispatch("linalg::csr_matrix::copy_upper_triangular_from",
                     host_extract_upper_triangular_nnz,
                     device_extract_upper_triangular_nnz,
                     A,
                     U,
                     nnz_U);

    std::cout << "nnz_U: " << nnz_U << std::endl;

    U.resize(m, n, nnz_U);

    backend_dispatch("linalg::ssor_fill_upper_precond",
                     host_ssor_fill_upper_precond,
                     device_ssor_fill_upper_precond,
                     A,
                     U,
                     beta);

    csrtrsv_analysis(this->L, triangular_type::lower, diagonal_type::non_unit, this->descr_L);
    csrtrsv_analysis(this->U, triangular_type::upper, diagonal_type::non_unit, this->descr_U);

    //y.resize(this->L.get_m());
}

void SSOR_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("SSOR_precond::solve");

    // Let alpha = 1 / (omega * (2-omega)) and beta = 1 / omega
    double alpha = 1.0 / (omega * (2.0 - omega));

    // M = alpha * (beta * D - E) * D^-1 * (beta * D - F), where A = D-E-F
    //
    // |*    |
    // | *-F |
    // |  D  |
    // |-E * |
    // |    *|
    //
    // Solve M * x = rhs
    // alpha * (beta * D - E) * D^-1 * (beta * D - F) * x = rhs
    // Let y = (alpha * (beta * D - E) * D^-1 * x and therefore
    // (alpha * (beta * D - E) * D^-1 * y = rhs
    // So solve (alpha * (beta * D - E) * D^-1 * y = rhs followed by (beta * D - F) * x = y

    // Let L = (beta * D - E) * D^-1 and U = (beta * D - F). This gives M = alpha * L * U.
    vector<double> y(this->L.get_m());

    if(rhs.is_on_host())
    {
        y.move_to_host();
    }
    else
    {
        y.move_to_device();
    }

    // Solve alpha * (beta * D - E) * D^-1 * y = alpha * L * y = rhs
    csrtrsv_solve(
        this->L, rhs, y, alpha, triangular_type::lower, diagonal_type::non_unit, this->descr_L);

    // Solve (beta * D - F) * x = U * x = y
    csrtrsv_solve(
        this->U, y, x, 1.0, triangular_type::upper, diagonal_type::non_unit, this->descr_U);
}

void SSOR_precond::move_to_device()
{
    // this->y.move_to_device();
    this->L.move_to_device();
    this->U.move_to_device();
    this->on_host = false;
}

void SSOR_precond::move_to_host()
{
    // this->y.move_to_host();
    this->L.move_to_host();
    this->U.move_to_host();
    this->on_host = true;
}

bool SSOR_precond::is_on_host() const
{
    return on_host;
}

ilu_precond::ilu_precond()
    : on_host(true)
{
    create_csrtrsv_descr(&descr_L);
    create_csrtrsv_descr(&descr_U);
}
ilu_precond::~ilu_precond()
{
    destroy_csrtrsv_descr(descr_L);
    destroy_csrtrsv_descr(descr_U);
}

void ilu_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("ilu_precond::build");

    this->LU.copy_from(A);

    int structural_zero = -1;
    int numeric_zero    = -1;

    // In place incomplete LU factorization
    csrilu0(LU, &structural_zero, &numeric_zero);

    // Analyze L and U factors for triangular solves
    csrtrsv_analysis(this->LU, triangular_type::lower, diagonal_type::unit, this->descr_L);
    csrtrsv_analysis(this->LU, triangular_type::upper, diagonal_type::non_unit, this->descr_U);
}

void ilu_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("ilu_precond::solve");

    // L * U * x = rhs
    // Let y = U * x
    vector<double> y(rhs.get_size());

    // Solve L * y = rhs
    csrtrsv_solve(LU, rhs, y, 1.0, triangular_type::lower, diagonal_type::unit, this->descr_L);

    // Solve U * x = y
    csrtrsv_solve(LU, y, x, 1.0, triangular_type::upper, diagonal_type::non_unit, this->descr_U);
}

void ilu_precond::move_to_device()
{
    this->LU.move_to_device();
    this->on_host = false;
}

void ilu_precond::move_to_host()
{
    this->LU.move_to_host();
    this->on_host = true;
}

bool ilu_precond::is_on_host() const
{
    return on_host;
}

ic_precond::ic_precond()
    : on_host(true)
{
    create_csrtrsv_descr(&descr_L);
    create_csrtrsv_descr(&descr_LT);
}
ic_precond::~ic_precond()
{
    destroy_csrtrsv_descr(descr_L);
    destroy_csrtrsv_descr(descr_LT);
}

void ic_precond::build(const csr_matrix& A)
{
    ROUTINE_TRACE("ic_precond::build");

    this->LLT.copy_from(A);

    int structural_zero = -1;
    int numeric_zero    = -1;

    int     m               = LLT.get_m();
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

    // Analyze L and L^T factors for triangular solves
    csrtrsv_analysis(this->LLT, triangular_type::lower, diagonal_type::non_unit, this->descr_L);
    csrtrsv_analysis(this->LLT, triangular_type::upper, diagonal_type::non_unit, this->descr_LT);
}

void ic_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    ROUTINE_TRACE("ic_precond::solve");

    // L * L^T * x = rhs
    // Let y = L^T * x
    vector<double> y(rhs.get_size());

    // Solve L * y = rhs
    csrtrsv_solve(LLT, rhs, y, 1.0, triangular_type::lower, diagonal_type::non_unit, this->descr_L);

    // Solve L^T * x = y
    csrtrsv_solve(LLT, y, x, 1.0, triangular_type::upper, diagonal_type::non_unit, this->descr_LT);
}

void ic_precond::move_to_device()
{
    this->LLT.move_to_device();
    this->on_host = false;
}

void ic_precond::move_to_host()
{
    this->LLT.move_to_host();
    this->on_host = true;
}

bool ic_precond::is_on_host() const
{
    return on_host;
}

itilu_precond::itilu_precond()
    : on_host(true)
{
}
itilu_precond::~itilu_precond() {}

void itilu_precond::build(const csr_matrix& A)
{

    // this->LU.copy_from(A);

    // int structural_zero = -1;
    // int numeric_zero    = -1;

    // // In place incomplete LU factorization
    // csrilu0(LU, &structural_zero, &numeric_zero);
}

void itilu_precond::solve(const vector<double>& rhs, vector<double>& x) const
{
    // // L * U * x = rhs
    // // Let y = U * x
    // vector<double> y(rhs.get_size());

    // // Solve L * y = rhs
    // forward_solve(LU, rhs, y, true);

    // // Solve U * x = y
    // backward_solve(LU, y, x, false);
}

void itilu_precond::move_to_device()
{
    this->on_host = false;
}

void itilu_precond::move_to_host()
{
    this->on_host = true;
}

bool itilu_precond::is_on_host() const
{
    return on_host;
}
