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

#include "../include/linalg_math.h"
#include <assert.h>
#include <iostream>

#include "trace.h"
#include "utility.h"

#include "backend/device/device_math.h"
#include "backend/host/host_math.h"

// Compute y = alpha * x + y
void linalg::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::axpy");

    backend_dispatch("linalg::axpy", host_axpy, device_axpy, alpha, x, y);
}

// Compute y = alpha * x + beta * y
void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::axpby");

    backend_dispatch("linalg::axpby", host_axpby, device_axpby, alpha, x, beta, y);
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::axpbypgz(double                alpha,
                      const vector<double>& x,
                      double                beta,
                      const vector<double>& y,
                      double                gamma,
                      vector<double>&       z)
{
    ROUTINE_TRACE("linalg::axpbypgz");

    backend_dispatch(
        "linalg::axpbypgz", host_axpbypgz, device_axpbypgz, alpha, x, beta, y, gamma, z);
}

// Incomplete IC factorization
void linalg::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csric0");

    backend_dispatch(
        "linalg::csric0", host_csric0, device_csric0, LL, structural_zero, numeric_zero);
}

// Incomplete LU factorization
void linalg::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csrilu0");

    backend_dispatch(
        "linalg::csrilu0", host_csrilu0, device_csrilu0, LU, structural_zero, numeric_zero);
}

// Transpose matrix
void linalg::transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::transpose_matrix");

    backend_dispatch(
        "linalg::transpose_matrix", host_transpose_matrix, device_transpose_matrix, A, transposeA);
}

// Dot product
double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::dot_product");

    return backend_dispatch("linalg::dot_product", host_dot_product, device_dot_product, x, y);
}

// Compute residual
void linalg::compute_residual(const csr_matrix&     A,
                              const vector<double>& x,
                              const vector<double>& b,
                              vector<double>&       res)
{
    ROUTINE_TRACE("linalg::compute_residual");

    backend_dispatch(
        "linalg::compute_residual", host_compute_residual, device_compute_residual, A, x, b, res);
}

// Extract diagonal entries
void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::diagonal");

    backend_dispatch("linalg::diagonal", host_diagonal, device_diagonal, A, d);
}

// Euclidean norm
double linalg::norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_euclid");

    return backend_dispatch("linalg::norm_euclid", host_norm_euclid, device_norm_euclid, array);
}

// Infinity norm
double linalg::norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_inf");

    return backend_dispatch("linalg::norm_inf", host_norm_inf, device_norm_inf, array);
}

struct linalg::csrtrsv_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::create_csrtrsv_descr(csrtrsv_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csrtrsv_descr");

    *descr = new csrtrsv_descr;
    allocate_csrtrsv_device_data(*descr);
}

void linalg::destroy_csrtrsv_descr(csrtrsv_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csrtrsv_descr");

    if(descr != nullptr)
    {
        free_csrtrsv_device_data(descr);

        delete descr;
    }
}

void linalg::csrtrsv_analysis(const csr_matrix& A,
                              triangular_type   tri_type,
                              diagonal_type     diag_type,
                              csrtrsv_descr*    descr)
{
    ROUTINE_TRACE("linalg::csrtrsv_analysis");

    return backend_dispatch("linalg::csrtrsv_analysis",
                            host_csrtrsv_analysis,
                            device_csrtrsv_analysis,
                            A,
                            tri_type,
                            diag_type,
                            descr);
}

void linalg::csrtrsv_solve(const csr_matrix&     A,
                           const vector<double>& b,
                           vector<double>&       x,
                           double                alpha,
                           triangular_type       tri_type,
                           diagonal_type         diag_type,
                           const csrtrsv_descr*  descr)
{
    ROUTINE_TRACE("linalg::csrtrsv_solve");

    return backend_dispatch("linalg::csrtrsv_solve",
                            host_csrtrsv_solve,
                            device_csrtrsv_solve,
                            A,
                            b,
                            x,
                            alpha,
                            tri_type,
                            diag_type,
                            descr);
}

struct linalg::csrmv_descr
{
};

void linalg::create_csrmv_descr(csrmv_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csrmv_descr");

    *descr = new csrmv_descr;
    allocate_csrmv_device_data(*descr);
}

void linalg::destroy_csrmv_descr(csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csrmv_descr");

    if(descr != nullptr)
    {
        free_csrmv_device_data(descr);

        delete descr;
    }
}

void linalg::csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::csrmv_analysis");

    return backend_dispatch(
        "linalg::csrmv_analysis", host_csrmv_analysis, device_csrmv_analysis, A, alg, descr);
}

void linalg::csrmv_solve(double                alpha,
                         const csr_matrix&     A,
                         const vector<double>& x,
                         double                beta,
                         vector<double>&       y,
                         csrmv_algorithm       alg,
                         const csrmv_descr*    descr)
{
    ROUTINE_TRACE("linalg::csrmv_solve");

    return backend_dispatch("linalg::csrmv_solve",
                            host_csrmv_solve,
                            device_csrmv_solve,
                            alpha,
                            A,
                            x,
                            beta,
                            y,
                            alg,
                            descr);
}

struct linalg::csrgeam_descr
{
};

void linalg::create_csrgeam_descr(csrgeam_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csrgeam_descr");

    *descr = new csrgeam_descr;
    allocate_csrgeam_device_data(*descr);
}
void linalg::destroy_csrgeam_descr(csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csrgeam_descr");

    if(descr != nullptr)
    {
        free_csrgeam_device_data(descr);

        delete descr;
    }
}

void linalg::csrgeam_nnz(const csr_matrix& A,
                         const csr_matrix& B,
                         csr_matrix&       C,
                         csrgeam_algorithm alg,
                         csrgeam_descr*    descr)
{
    ROUTINE_TRACE("linalg::csrgeam_nnz");

    return backend_dispatch(
        "linalg::csrgeam_nnz", host_csrgeam_nnz, device_csrgeam_nnz, A, B, C, alg, descr);
}

void linalg::csrgeam_solve(double               alpha,
                           const csr_matrix&    A,
                           double               beta,
                           const csr_matrix&    B,
                           csr_matrix&          C,
                           csrgeam_algorithm    alg,
                           const csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::csrgeam_solve");

    return backend_dispatch("linalg::csrgeam_solve",
                            host_csrgeam_solve,
                            device_csrgeam_solve,
                            alpha,
                            A,
                            beta,
                            B,
                            C,
                            alg,
                            descr);
}

struct linalg::csrgemm_descr
{
};

void linalg::create_csrgemm_descr(csrgemm_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csrgemm_descr");

    *descr = new csrgemm_descr;
    allocate_csrgemm_device_data(*descr);
}

void linalg::destroy_csrgemm_descr(csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csrgemm_descr");

    if(descr != nullptr)
    {
        free_csrgemm_device_data(descr);

        delete descr;
    }
}

void linalg::csrgemm_nnz(const csr_matrix& A,
                         const csr_matrix& B,
                         const csr_matrix& D,
                         csr_matrix&       C,
                         csrgemm_algorithm alg,
                         csrgemm_descr*    descr)
{
    ROUTINE_TRACE("linalg::csrgeam_nnz");

    return backend_dispatch(
        "linalg::csrgemm_nnz", host_csrgemm_nnz, device_csrgemm_nnz, A, B, D, C, alg, descr);
}
void linalg::csrgemm_solve(double               alpha,
                           const csr_matrix&    A,
                           const csr_matrix&    B,
                           double               beta,
                           const csr_matrix&    D,
                           csr_matrix&          C,
                           csrgemm_algorithm    alg,
                           const csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::csrgemm_solve");

    return backend_dispatch("linalg::csrgemm_solve",
                            host_csrgemm_solve,
                            device_csrgemm_solve,
                            alpha,
                            A,
                            B,
                            beta,
                            D,
                            C,
                            alg,
                            descr);
}

struct linalg::csric0_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::create_csric0_descr(csric0_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csric0_descr");

    *descr = new csric0_descr;
    allocate_csric0_device_data(*descr);
}

void linalg::destroy_csric0_descr(csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csric0_descr");

    if(descr != nullptr)
    {
        free_csric0_device_data(descr);

        delete descr;
    }
}

void linalg::csric0_analysis(const csr_matrix& A, csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::csric0_analysis");

    return backend_dispatch(
        "linalg::csric0_analysis", host_csric0_analysis, device_csric0_analysis, A, descr);
}

void linalg::csric0_compute(csr_matrix& A, const csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::csric0_compute");

    return backend_dispatch(
        "linalg::csric0_compute", host_csric0_compute, device_csric0_compute, A, descr);
}

struct linalg::csrilu0_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::create_csrilu0_descr(csrilu0_descr** descr)
{
    ROUTINE_TRACE("linalg::create_csrilu0_descr");

    *descr = new csrilu0_descr;
    allocate_csrilu0_device_data(*descr);
}

void linalg::destroy_csrilu0_descr(csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_csrilu0_descr");

    if(descr != nullptr)
    {
        free_csrilu0_device_data(descr);
        delete descr;
    }
}

void linalg::csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::csrilu0_analysis");

    return backend_dispatch(
        "linalg::csrilu0_analysis", host_csrilu0_analysis, device_csrilu0_analysis, A, descr);
}

void linalg::csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::csrilu0_compute");

    return backend_dispatch(
        "linalg::csrilu0_compute", host_csrilu0_compute, device_csrilu0_compute, A, descr);
}

void linalg::tridiagonal_solver(int                   m,
                                int                   n,
                                const vector<double>& lower_diag,
                                const vector<double>& main_diag,
                                const vector<double>& upper_diag,
                                const vector<double>& rhs,
                                vector<double>&       solution)
{
    ROUTINE_TRACE("linalg::tridiagonal_solver");

    return backend_dispatch("linalg::tridiagonal_solver",
                            host_tridiagonal_solver,
                            device_tridiagonal_solver,
                            m,
                            n,
                            lower_diag,
                            main_diag,
                            upper_diag,
                            rhs,
                            solution);
}
