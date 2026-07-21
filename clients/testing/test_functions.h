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

#ifndef TEST_FUNCTIONS_H__
#define TEST_FUNCTIONS_H__

#include "test_arguments.h"

namespace testing
{
    bool test_dispatch(Arguments arg);

    // iterative solvers
    bool test_classical(classical_solver solver_type, Arguments arg);
    bool test_krylov(krylov_solver solver_type, Arguments arg);
    bool test_amg(AMG_solver solver_type, Arguments arg);

    // direct solvers
    bool test_tridiagonal_solver(Arguments arg);

    // math testing
    bool test_sptrsv(Arguments arg);
    bool test_spgeam(Arguments arg);
    bool test_csric0(Arguments arg);
    bool test_csrilu0(Arguments arg);

    // primitive
    bool test_exclusive_scan(Arguments arg);

    // csr matrix
    bool test_transpose(Arguments arg);
    bool test_transpose_dense(Arguments arg);
    bool test_multiply_by_vector(Arguments arg);
    bool test_multiply_by_matrix(Arguments arg);
    bool test_triangular_solve(Arguments arg);
    bool test_compute_incomplete_cholesky_factorization(Arguments arg);
    bool test_compute_incomplete_cholesky_factorization_dense(Arguments arg);
    bool test_compute_incomplete_LU_factorization_dense(Arguments arg);

} // namespace testing

#endif
