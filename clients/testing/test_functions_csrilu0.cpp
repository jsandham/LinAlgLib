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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "linalg.h"

bool testing::test_csrilu0(Arguments arg)
{
    const int                              m   = arg.m;
    const double                           tol = arg.tol;
    std::mt19937                           gen(123456);
    const int                              max_bandwidth = std::min(m, 32);
    std::uniform_int_distribution<int>     uniform_band_width_dist(1, max_bandwidth);
    std::normal_distribution<double>       normal_band_width_dist((max_bandwidth + 1) / 2.0,
                                                            std::max(1.0, max_bandwidth / 6.0));
    std::uniform_real_distribution<double> uniform_value_dist(-1.0, 1.0);
    std::normal_distribution<double>       normal_value_dist(0.0, 1.0);
    std::vector<int>                       first_col_L(m);
    std::vector<int>                       last_col_U(m);

    auto sample_band_width = [&]() {
        switch(arg.distribution)
        {
        case testing::data_distribution::uniform:
            return uniform_band_width_dist(gen);
        case testing::data_distribution::normal:
            return std::clamp(
                static_cast<int>(std::lround(normal_band_width_dist(gen))), 1, max_bandwidth);
        }

        return 1;
    };

    auto sample_off_diagonal = [&]() {
        switch(arg.distribution)
        {
        case testing::data_distribution::uniform:
            return uniform_value_dist(gen);
        case testing::data_distribution::normal:
            return normal_value_dist(gen);
        }

        return 0.0;
    };

    std::cout << "m: " << m << " tol: " << tol
              << " distribution: " << data_distribution_to_string(arg.distribution) << std::endl;

    int previous_first_col = 0;
    int previous_last_col  = 0;
    for(int i = 0; i < m; i++)
    {
        const int lower_width = std::min(i + 1, sample_band_width());
        first_col_L[i]        = std::max(previous_first_col, i + 1 - lower_width);
        previous_first_col    = first_col_L[i];

        const int upper_width = sample_band_width() - 1;
        last_col_U[i]         = std::max(previous_last_col, std::min(m - 1, i + upper_width));
        previous_last_col     = last_col_U[i];
    }

    int nnz_L = 0;
    int nnz_U = 0;
    for(int i = 0; i < m; i++)
    {
        nnz_L += i - first_col_L[i] + 1;
        nnz_U += last_col_U[i] - i + 1;
    }

    std::vector<int>    csr_row_ptr_L(m + 1);
    std::vector<int>    csr_col_ind_L(nnz_L);
    std::vector<double> csr_val_L(nnz_L);
    std::vector<int>    csr_row_ptr_U(m + 1);
    std::vector<int>    csr_col_ind_U(nnz_U);
    std::vector<double> csr_val_U(nnz_U);

    csr_row_ptr_L[0] = 0;
    csr_row_ptr_U[0] = 0;
    for(int i = 0; i < m; i++)
    {
        const int lower_off_diagonal_count = i - first_col_L[i];
        double    lower_off_diagonal_sum   = 0.0;
        int       index_L                  = csr_row_ptr_L[i];
        for(int col = first_col_L[i]; col < i; col++)
        {
            csr_col_ind_L[index_L] = col;
            csr_val_L[index_L]     = sample_off_diagonal();
            lower_off_diagonal_sum += std::abs(csr_val_L[index_L]);
            index_L++;
        }
        const double lower_value_scale
            = lower_off_diagonal_sum > 0.0 ? 0.5 / lower_off_diagonal_sum : 0.0;
        for(int j = csr_row_ptr_L[i]; j < index_L; j++)
        {
            csr_val_L[j] *= lower_value_scale;
        }
        csr_col_ind_L[index_L] = i;
        csr_val_L[index_L]     = 1.0;
        csr_row_ptr_L[i + 1]   = index_L + 1;

        double upper_off_diagonal_sum = 0.0;
        int    index_U                = csr_row_ptr_U[i];
        for(int col = i; col <= last_col_U[i]; col++)
        {
            csr_col_ind_U[index_U] = col;
            if(col == i)
            {
                csr_val_U[index_U] = 1.0;
            }
            else
            {
                csr_val_U[index_U] = sample_off_diagonal();
                upper_off_diagonal_sum += std::abs(csr_val_U[index_U]);
            }
            index_U++;
        }
        const double upper_value_scale
            = upper_off_diagonal_sum > 0.0 ? 0.5 / upper_off_diagonal_sum : 0.0;
        for(int j = csr_row_ptr_U[i] + 1; j < index_U; j++)
        {
            csr_val_U[j] *= upper_value_scale;
        }
        upper_off_diagonal_sum *= upper_value_scale;
        csr_val_U[csr_row_ptr_U[i]] = upper_off_diagonal_sum + 1.0;
        csr_row_ptr_U[i + 1]        = index_U;
    }

    linalg::csr_matrix<double> mat_L(csr_row_ptr_L, csr_col_ind_L, csr_val_L, m, m, nnz_L);
    linalg::csr_matrix<double> mat_U(csr_row_ptr_U, csr_col_ind_U, csr_val_U, m, m, nnz_U);

    linalg::csr_matrix<double> mat_A;
    mat_L.multiply_by_matrix(mat_A, mat_U);

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
    }

    linalg::csrilu0_descr* descr = nullptr;
    linalg::create_csrilu0_descr(&descr);
    linalg::csrilu0_analysis(mat_A, descr);

    auto t1 = std::chrono::high_resolution_clock::now();
    linalg::csrilu0_compute(mat_A, descr);
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = t2 - t1;
    std::cout << "csrilu0: " << ms.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
    }

    linalg::csr_matrix<double> mat_L_solution;
    linalg::csr_matrix<double> mat_U_solution;
    mat_L_solution.copy_lower_triangular_from(mat_A, false);
    mat_U_solution.copy_upper_triangular_from(mat_A, false);

    for(int i = 0; i < m; i++)
    {
        for(int j = mat_L_solution.get_row_ptr()[i]; j < mat_L_solution.get_row_ptr()[i + 1]; j++)
        {
            if(mat_L_solution.get_col_ind()[j] == i)
            {
                mat_L_solution.get_val()[j] = 1.0;
                break;
            }
        }
    }

    linalg::destroy_csrilu0_descr(descr);

    const bool lower_matches = check_matrix_equality(mat_L, mat_L_solution, tol);
    const bool upper_matches = check_matrix_equality(mat_U, mat_U_solution, tol);
    return lower_matches && upper_matches;
}
