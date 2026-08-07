//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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
#include "utility.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "linalg.h"

bool testing::test_compute_incomplete_cholesky_factorization_dense(Arguments arg)
{
    // Create a dense SPD matrix.
    std::vector<int> csr_row_ptr(arg.m + 1, 0);
    for(int i = 0; i < arg.m; i++)
    {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + arg.m;
    }

    std::vector<int>    csr_col_ind(arg.m * arg.m, 0);
    std::vector<double> csr_val(arg.m * arg.m, 0.0);
    for(int i = 0; i < arg.m; i++)
    {
        for(int j = 0; j < arg.m; j++)
        {
            csr_col_ind[arg.m * i + j] = j;
            csr_val[arg.m * i + j]     = (i == j) ? 2.0 : 1.0; // Create a simple SPD matrix
        }
    }

    linalg::csr_matrix<double> mat_A(csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     arg.m,
                                     arg.m,
                                     arg.m * arg.m); // Assuming a dense matrix for testing
    mat_A.make_diagonally_dominant();

    linalg::vector<double> ones(mat_A.get_n());
    ones.ones();

    linalg::vector<double> b(mat_A.get_m());
    b.zeros();

    linalg::vector<double> y(mat_A.get_n());
    y.zeros();

    linalg::vector<double> z(mat_A.get_n());
    z.zeros();

    linalg::csr_matrix<double> mat_transpose;
    mat_transpose.resize(mat_A.get_n(), mat_A.get_m(), mat_A.get_nnz());

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_transpose.move_to_device();
        b.move_to_device();
        y.move_to_device();
        z.move_to_device();
        ones.move_to_device();
    }

    mat_A.multiply_by_vector(b, ones);

    // LL^T * x = b
    // Let y = L^T * z and b = L * y
    // Step 1: Solve L * y = b
    // Step 2: Solve L^T * z = y
    // Step 3: Verify that z is approximately equal to ones
    mat_A.compute_incomplete_cholesky_factorization();

    // Solve L * y = b
    mat_A.triangular_solve_lower(y, b, false);

    // Solve L^T * z = y
    mat_A.transpose(mat_transpose);
    mat_transpose.triangular_solve_upper(z, y, false);

    if(arg.backend == backend::GPU)
    {
        z.move_to_host();
        ones.move_to_host();
    }

    // Verify that z is approximately equal to ones
    bool success = check_vector_equality(z, ones);
    if(!success)
    {
        std::cout << "Cholesky result does not match reference solution." << std::endl;
    }

    return success;
}
