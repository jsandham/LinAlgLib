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

#include <chrono>
#include <cmath>
#include <iostream>

#include "linalg.h"

bool testing::test_spgemm(Arguments arg)
{
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
        mat_C.move_to_host();
        // mat_C.print_matrix("C");
    }

    // Inline host solution
    linalg::csr_matrix mat_C_host;

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
