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
#include "utility.h"

#include <chrono>
#include <cmath>
#include <iostream>

#include "linalg.h"

bool testing::test_transpose_dense(Arguments arg)
{
    // Create a dense matrix.
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
            csr_val[arg.m * i + j]     = static_cast<double>(i * arg.m + j + 1);
        }
    }

    linalg::csr_matrix mat_A(csr_row_ptr,
                             csr_col_ind,
                             csr_val,
                             arg.m,
                             arg.m,
                             arg.m * arg.m); // Assuming a dense matrix for testing

    linalg::csr_matrix mat_A_transpose;
    mat_A_transpose.resize(mat_A.get_n(), mat_A.get_m(), mat_A.get_nnz());

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_A_transpose.move_to_device();
    }

    mat_A.transpose(mat_A_transpose);

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        mat_A_transpose.move_to_host();
    }

    //mat_A.print_matrix("Original Matrix A");
    //mat_A_transpose.print_matrix("Transposed Matrix A^T");

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

    return success;
}
