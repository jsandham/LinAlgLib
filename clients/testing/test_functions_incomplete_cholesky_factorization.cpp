//********************************************************************************
//
// MIT License
//
// Copyright(c) 202-2026 James Sandham
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

static bool compute_reference_incomplete_cholesky(const linalg::csr_matrix<double>& A,
                                                  linalg::csr_matrix<double>&       L)
{
    L.copy_from(A);

    int m = L.get_m();
    int n = L.get_n();
    if(m != n)
    {
        return false;
    }

    const int* row_ptr = L.get_row_ptr();
    const int* col_ind = L.get_col_ind();
    double*    val     = L.get_val();

    std::vector<int> diag_ptr(m, -1);
    std::vector<int> col_offset(n, -1);

    for(int row = 0; row < m; row++)
    {
        int row_begin = row_ptr[row];
        int row_end   = row_ptr[row + 1];

        std::fill(col_offset.begin(), col_offset.end(), -1);
        for(int j = row_begin; j < row_end; j++)
        {
            col_offset[col_ind[j]] = j;
        }

        double sum = 0.0;
        for(int j = row_begin; j < row_end; j++)
        {
            int col_j = col_ind[j];
            if(col_j < row)
            {
                int    diag_index = diag_ptr[col_j];
                double s          = 0.0;
                for(int k = row_ptr[col_j]; k < diag_index; k++)
                {
                    int col_k = col_ind[k];
                    int pos   = col_offset[col_k];
                    if(pos != -1)
                    {
                        s += val[pos] * val[k];
                    }
                }

                double diag_val = 1.0;
                if(diag_index != -1)
                {
                    diag_val = val[diag_index];
                    if(diag_val == 0.0)
                    {
                        diag_val = 1.0;
                    }
                }

                val[j] = (val[j] - s) / diag_val;
                sum += val[j] * val[j];
            }
            else if(col_j == row)
            {
                diag_ptr[row]   = j;
                double diag_val = std::max(0.0, val[j] - sum);
                val[j]          = std::sqrt(diag_val);
                break;
            }
            else
            {
                break;
            }
        }
    }

    return true;
}

bool testing::test_compute_incomplete_cholesky_factorization(Arguments arg)
{
    linalg::csr_matrix<double> mat_A;
    mat_A.read_mtx(arg.filename);

    if(mat_A.get_m() != mat_A.get_n())
    {
        std::cerr << "Matrix must be square for Cholesky factorization." << std::endl;
        return false;
    }

    linalg::csr_matrix<double> mat_A_copy;
    mat_A_copy.copy_from(mat_A);

    linalg::csr_matrix<double> mat_A_reference;
    if(!compute_reference_incomplete_cholesky(mat_A, mat_A_reference))
    {
        std::cerr << "Failed to compute reference Cholesky factorization." << std::endl;
        return false;
    }

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_A_copy.move_to_device();
    }

    // Warmup
    for(int i = 0; i < 1; i++)
    {
        mat_A.copy_from(mat_A_copy);
        mat_A.compute_incomplete_cholesky_factorization();
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++)
    {
        mat_A.copy_from(mat_A_copy);
        mat_A.compute_incomplete_cholesky_factorization();
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Cholesky compute time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
    }

    mat_A.print_matrix("Computed Cholesky Factorization");
    mat_A_reference.print_matrix("Reference Cholesky Factorization");

    bool success = check_matrix_equality(mat_A, mat_A_reference);
    if(!success)
    {
        std::cout << "Cholesky result does not match reference solution." << std::endl;
    }

    return success;
}
