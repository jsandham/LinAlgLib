//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#include "test_functions.h"

static std::string correct_filename(const std::string& filename)
{
#if defined(_WIN32) || defined(WIN32)
    return "../" + filename;
#else
    return filename;
#endif
}

namespace testing
{
    static bool iterative_solver_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
        case fixture::jacobi:
            return test_classical(classical_solver::jacobi, arg);
        case fixture::gauss_seidel:
            return test_classical(classical_solver::gauss_seidel, arg);
        case fixture::SOR:
            return test_classical(classical_solver::SOR, arg);
        case fixture::symmetric_gauss_seidel:
            return test_classical(classical_solver::symmetric_gauss_seidel, arg);
        case fixture::SSOR:
            return test_classical(classical_solver::SSOR, arg);
        case fixture::CG:
            return test_krylov(krylov_solver::CG, arg);
        case fixture::BICGSTAB:
            return test_krylov(krylov_solver::BICGSTAB, arg);
        case fixture::GMRES:
            return test_krylov(krylov_solver::GMRES, arg);
        case fixture::UAAMG:
            return test_amg(AMG_solver::UAAMG, arg);
        case fixture::SAAMG:
            return test_amg(AMG_solver::SAAMG, arg);
        case fixture::RSAMG:
            return test_amg(AMG_solver::RSAMG, arg);
        }

        return false;
    }

    static bool direct_solver_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
        case fixture::tridiagonal_solver:
            return test_tridiagonal_solver(arg);
        }

        return false;
    }

    static bool math_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
        case fixture::axpy:
            return test_axpy(arg);
        case fixture::axpby:
            return test_axpby(arg);
        case fixture::axpbypgz:
            return test_axpbypgz(arg);
        case fixture::SpTRSV:
            return test_sptrsv(arg);
        case fixture::SpGEAM:
            return test_spgeam(arg);
        case fixture::CSRIC0:
            return test_csric0(arg);
        case fixture::CSRILU0:
            return test_csrilu0(arg);
        }

        return false;
    }

    static bool primitive_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
        case fixture::exclusive_scan:
            return test_exclusive_scan(arg);
        }

        return false;
    }

    static bool csr_matrix_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
        case fixture::multiply_by_vector:
            return test_multiply_by_vector(arg);
        case fixture::multiply_by_matrix:
            return test_multiply_by_matrix(arg);
        case fixture::triangular_solve:
            return test_triangular_solve(arg);
        case fixture::compute_incomplete_cholesky_factorization:
            return test_compute_incomplete_cholesky_factorization(arg);
        case fixture::compute_incomplete_cholesky_factorization_dense:
            return test_compute_incomplete_cholesky_factorization_dense(arg);
        case fixture::compute_incomplete_LU_factorization_dense:
            return test_compute_incomplete_LU_factorization_dense(arg);
        case fixture::transpose:
            return test_transpose(arg);
        case fixture::transpose_dense:
            return test_transpose_dense(arg);
        case fixture::ruiz_scaling:
            return test_ruiz_scaling(arg);
        case fixture::symmetric_ruiz_scaling:
            return test_symmetric_ruiz_scaling(arg);
        }
        return false;
    }
}

bool testing::test_dispatch(Arguments arg)
{
    arg.filename = correct_filename(arg.filename);

    if(arg.backend == backend::GPU)
    {
        if(!linalg::is_device_available())
        {
            std::cout << "Error: GPU backend requested but no device is available. Skipping test."
                      << std::endl;
            return true; // Skip the test gracefully
        }
    }

    std::cout << "category: " << category_to_string(arg.category) << std::endl;
    std::cout << "fixture: " << fixture_to_string(arg.fixture) << std::endl;

    switch(arg.category)
    {
    case category::iterative_solvers:
        return iterative_solver_test_dispatch(arg);
    case category::direct_solvers:
        return direct_solver_test_dispatch(arg);
    case category::math:
        return math_test_dispatch(arg);
    case category::primitive:
        return primitive_test_dispatch(arg);
    case category::csr_matrix:
        return csr_matrix_test_dispatch(arg);
    }

    return false;
}
