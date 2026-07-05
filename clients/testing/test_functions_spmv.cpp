//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

// Should rename this test to test_functions_multiply_by_vector.
// Should rename test_functions_spgemm.cpp to test_functions_multiply_by_matrix.cpp

bool testing::test_spmv(Arguments arg)
{
    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    std::mt19937                           generator(1234567);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for(int i = 0; i < mat_A.get_nnz(); i++)
    {
        mat_A.get_val()[i] = distribution(generator);
    }

    linalg::vector<double> vec_x(mat_A.get_n());
    vec_x.ones();

    linalg::vector<double> vec_y(mat_A.get_m());
    vec_y.zeros();

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        vec_x.move_to_device();
        vec_y.move_to_device();
    }

    // Warmup
    for(int i = 0; i < 4; i++)
    {
        mat_A.multiply_by_vector(vec_y, vec_x);
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        mat_A.multiply_by_vector(vec_y, vec_x);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        vec_x.move_to_host();
        vec_y.move_to_host();
    }

    // Inline host solution
    linalg::vector<double> host_y(mat_A.get_m());
    host_y.zeros();
    for(int i = 0; i < mat_A.get_m(); ++i)
    {
        for(int j = mat_A.get_row_ptr()[i]; j < mat_A.get_row_ptr()[i + 1]; ++j)
        {
            host_y[i] += mat_A.get_val()[j] * vec_x.get_vec()[mat_A.get_col_ind()[j]];
        }
    }

    // Compare solutions
    bool success = check_vector_equality(vec_y, host_y);

    if(!success)
    {
        std::cout << "SPMV test failed" << std::endl;
    }

    size_t total_bytes_read_write
        = sizeof(double) * (mat_A.get_nnz() + mat_A.get_m() + mat_A.get_n())
          + sizeof(int) * (mat_A.get_m() + 1 + mat_A.get_nnz());
    double total_gbytes = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth    = total_gbytes / (ms_float.count() / 1e3);
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return success;
}
