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

#include "test_functions.h"
#include "utility.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "linalg.h"

static linalg::csr_matrix compute_reference_spgemm(const linalg::csr_matrix& A,
                                                   const linalg::csr_matrix& B)
{
    int m = A.get_m();
    int n = B.get_n();

    const int*    row_ptr_A = A.get_row_ptr();
    const int*    col_ind_A = A.get_col_ind();
    const double* val_A     = A.get_val();

    const int*    row_ptr_B = B.get_row_ptr();
    const int*    col_ind_B = B.get_col_ind();
    const double* val_B     = B.get_val();

    // Step 1: Compute row pointers and nnz for C
    std::vector<int> row_ptr_C(m + 1, 0);
    {
        std::vector<int> nnz_marker(n, -1);
        for(int i = 0; i < m; ++i)
        {
            for(int j = row_ptr_A[i]; j < row_ptr_A[i + 1]; j++)
            {
                int col_A = col_ind_A[j];
                for(int p = row_ptr_B[col_A]; p < row_ptr_B[col_A + 1]; p++)
                {
                    int col_B = col_ind_B[p];
                    if(nnz_marker[col_B] != i)
                    {
                        nnz_marker[col_B] = i;
                        row_ptr_C[i + 1]++;
                    }
                }
            }
        }
        for(int i = 0; i < m; i++)
            row_ptr_C[i + 1] += row_ptr_C[i];
    }

    int nnz = row_ptr_C[m];

    // Step 2: Fill column indices and values for C
    std::vector<int>    col_ind_C(nnz);
    std::vector<double> val_C(nnz, 0.0);
    {
        std::vector<int> nnzs(n, -1);
        for(int i = 0; i < m; i++)
        {
            int row_begin_C = row_ptr_C[i];
            int row_end_C   = row_begin_C;
            for(int j = row_ptr_A[i]; j < row_ptr_A[i + 1]; j++)
            {
                int    col_A = col_ind_A[j];
                double val_a = val_A[j];
                for(int p = row_ptr_B[col_A]; p < row_ptr_B[col_A + 1]; p++)
                {
                    int    col_B = col_ind_B[p];
                    double val_b = val_B[p];
                    if(nnzs[col_B] < row_begin_C)
                    {
                        nnzs[col_B]          = row_end_C;
                        col_ind_C[row_end_C] = col_B;
                        val_C[row_end_C]     = val_a * val_b;
                        row_end_C++;
                    }
                    else
                    {
                        val_C[nnzs[col_B]] += val_a * val_b;
                    }
                }
            }
        }

        // Step 3: Sort columns within each row
        std::vector<int>    cols_copy(col_ind_C);
        std::vector<double> vals_copy(val_C);
        for(int i = 0; i < m; i++)
        {
            int              row_begin = row_ptr_C[i];
            int              row_nnz   = row_ptr_C[i + 1] - row_begin;
            std::vector<int> perm(row_nnz);
            for(int j = 0; j < row_nnz; j++)
                perm[j] = j;
            int*    col_entry = cols_copy.data() + row_begin;
            double* val_entry = vals_copy.data() + row_begin;
            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                return col_entry[a] < col_entry[b];
            });
            for(int j = 0; j < row_nnz; j++)
            {
                col_ind_C[row_begin + j] = col_entry[perm[j]];
                val_C[row_begin + j]     = val_entry[perm[j]];
            }
        }
    }

    return linalg::csr_matrix(row_ptr_C, col_ind_C, val_C, m, n, nnz);
}

bool testing::test_multiply_by_matrix(Arguments arg)
{
    std::cout << "Testing multiply_by_matrix with arguments: " << std::endl;

    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    for(int i = 0; i < mat_A.get_nnz(); i++)
    {
        double* csr_val = mat_A.get_val();
        csr_val[i]      = 1;
    }

    linalg::csr_matrix mat_B;
    mat_B.copy_from(mat_A);

    linalg::csr_matrix mat_C;
    mat_C.resize(mat_A.get_m(), mat_B.get_n(), 0);

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_B.move_to_device();
        mat_C.move_to_device();
    }

    // Warm up
    for(int i = 0; i < 4; i++)
    {
        mat_A.multiply_by_matrix(mat_C, mat_B);
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        mat_A.multiply_by_matrix(mat_C, mat_B);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        mat_B.move_to_host();
        mat_C.move_to_host();
    }

    // mat_C.print_matrix("C");

    // Inline host solution
    linalg::csr_matrix mat_C_host = compute_reference_spgemm(mat_A, mat_B);

    // mat_C_host.print_matrix("C_host");

    // Verify result
    bool success = check_matrix_equality(mat_C, mat_C_host);

    size_t bytes_read_A
        = sizeof(double) * mat_A.get_nnz() + sizeof(int) * (mat_A.get_m() + 1 + mat_A.get_nnz());
    size_t bytes_read_B
        = sizeof(double) * mat_B.get_nnz() + sizeof(int) * (mat_B.get_m() + 1 + mat_B.get_nnz());
    size_t bytes_write_C
        = sizeof(double) * mat_C.get_nnz() + sizeof(int) * (mat_C.get_m() + 1 + mat_C.get_nnz());

    size_t total_bytes_read_write = bytes_read_A + bytes_read_B + bytes_write_C;

    double total_gbytes = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth    = total_gbytes / (ms_float.count() / 1e3);
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return success;
}
