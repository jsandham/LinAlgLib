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
#include <random>

#include "linalg.h"

bool testing::test_csric0(Arguments arg)
{
    // Contruct a lower triangular "skyline" matrix L with positive diagonal entries.
    // We will then form A = L * L^T and compute Cholesky factorization on A. Because
    // L is a "skyline" matrix with positive diagonal entries, the Cholesky factorization
    // of A should not encounter any zero pivots. To ensure numerical stability, we ensure
    // diagonal dominance in L. After the Cholesky factorization of A is computed
    // inlplace, the lower part of A should equalthe original L.
    int    m   = arg.m;
    double tol = arg.tol;

    std::cout << "m: " << m << " tol: " << tol
              << " distribution: " << data_distribution_to_string(arg.distribution) << std::endl;

    std::mt19937 gen(123456);

    std::uniform_int_distribution<int> uniform_dist(1, m);
    std::normal_distribution<double>   normal_dist(1.0, 2.0);

    std::vector<int> nnz_per_row(m);
    for(int i = 0; i < m; i++)
    {
        switch(arg.distribution)
        {
        case testing::data_distribution::uniform:
            nnz_per_row[i] = std::min(i + 1, uniform_dist(gen));
            break;
        case testing::data_distribution::normal:
            nnz_per_row[i]
                = std::max(1, std::min(i + 1, (int)std::abs(std::lround(normal_dist(gen)))));
            break;
        }
    }

    std::vector<int> csr_row_ptr(m + 1);
    csr_row_ptr[0] = 0;
    for(int i = 0; i < m; i++)
    {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + nnz_per_row[i];
    }

    int nnz = csr_row_ptr[m];

    std::vector<int>                       csr_col_ind(nnz);
    std::vector<double>                    csr_val(nnz);
    std::uniform_real_distribution<double> val_dist(-0.5, 0.5);

    for(int i = 0; i < m; i++)
    {
        const int row_start = csr_row_ptr[i];
        const int row_end   = csr_row_ptr[i + 1];

        const int first_col = (i + 1) - nnz_per_row[i];
        const int last_col  = i + 1;

        double off_diag_sum = 0.0;
        for(int j = row_start; j < row_end; j++)
        {
            csr_col_ind[j] = (j - row_start) + first_col;
            if(j == (row_end - 1))
            {
                csr_val[j] = off_diag_sum + 1.0;
            }
            else
            {
                csr_val[j] = val_dist(gen);
                off_diag_sum += std::abs(csr_val[j]);
            }
        }
    }

    linalg::csr_matrix<double> mat_L(csr_row_ptr, csr_col_ind, csr_val, m, m, nnz);

    // Compute transpose L^T
    linalg::csr_matrix<double> mat_LT;
    mat_L.transpose(mat_LT);

    // Form A = L * L^T
    linalg::csr_matrix<double> mat_A;
    mat_L.multiply_by_matrix(mat_A, mat_LT);

    if(arg.backend == backend::GPU)
    {
        mat_L.move_to_device();
        mat_LT.move_to_device();
        mat_A.move_to_device();
    }

    // Prepare for csric0 analysis
    linalg::csric0_descr* descr = nullptr;
    linalg::create_csric0_descr(&descr);

    // Perform analysis
    linalg::csric0_analysis(mat_A, descr);

    // Compute csric0 on the host
    auto t1 = std::chrono::high_resolution_clock::now();
    linalg::csric0_compute(mat_A, descr);
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = t2 - t1;
    std::cout << "csric0: " << ms.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        mat_L.move_to_host();
    }

    linalg::csr_matrix<double> mat_L_solution;
    mat_L_solution.copy_lower_triangular_from(mat_A, false);

    linalg::destroy_csric0_descr(descr);

    return check_matrix_equality(mat_L, mat_L_solution, tol);
}
