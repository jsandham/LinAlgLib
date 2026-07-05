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

#include "linalg.h"

bool testing::test_transpose(Arguments arg)
{
    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    linalg::csr_matrix mat_A_transpose;

    mat_A_transpose.resize(mat_A.get_n(), mat_A.get_m(), mat_A.get_nnz());

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_A_transpose.move_to_device();
    }

    // Warmup
    for(int i = 0; i < 4; i++)
    {
        mat_A.transpose(mat_A_transpose);
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        mat_A.transpose(mat_A_transpose);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        mat_A_transpose.move_to_host();
    }

    bool success = true;

    // Verify transpose
    for(int i = 0; i < mat_A.get_m(); ++i)
    {
        for(int j = mat_A.get_row_ptr()[i]; j < mat_A.get_row_ptr()[i + 1]; ++j)
        {
            int    col = mat_A.get_col_ind()[j];
            double val = mat_A.get_val()[j];

            bool found = false;
            for(int k = mat_A_transpose.get_row_ptr()[col];
                k < mat_A_transpose.get_row_ptr()[col + 1];
                ++k)
            {
                if(mat_A_transpose.get_col_ind()[k] == i)
                {
                    if(std::abs(mat_A_transpose.get_val()[k] - val) < 1e-9)
                    {
                        found = true;
                    }
                    break;
                }
            }
            if(!found)
            {
                success = false;
                break;
            }
        }
        if(!success)
        {
            break;
        }
    }

    size_t total_bytes_read
        = sizeof(double) * mat_A.get_nnz() + sizeof(int) * (mat_A.get_m() + 1 + mat_A.get_nnz());
    size_t total_bytes_write
        = sizeof(double) * mat_A_transpose.get_nnz()
          + sizeof(int) * (mat_A_transpose.get_m() + 1 + mat_A_transpose.get_nnz());

    double total_gbytes = (double)100 * (total_bytes_read + total_bytes_write) / 1e9;
    double bandwidth    = total_gbytes / (ms_float.count() / 1e3);
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return success;
}
