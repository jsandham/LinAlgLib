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
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include "linalg.h"

bool testing::test_symmetric_ruiz_scaling(Arguments arg)
{
    std::cout << "max_iters: " << arg.max_iters << " tol: " << arg.tol << std::endl;

    linalg::csr_matrix<double> mat_A;
    mat_A.read_mtx(arg.filename);

    linalg::vector<double> D(mat_A.get_m());

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        D.move_to_device();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    mat_A.apply_symmetric_ruiz_scaling(D, arg.max_iters, arg.tol);
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        D.move_to_host();
    }

    // Verify that the scaled matrix has row and column norms close to 1
    std::vector<double> row_norm(mat_A.get_m(), 0.0);

    for(int i = 0; i < mat_A.get_m(); i++)
    {
        const int start = mat_A.get_row_ptr()[i];
        const int end   = mat_A.get_row_ptr()[i + 1];

        for(int j = start; j < end; j++)
        {
            const int    col = mat_A.get_col_ind()[j];
            const double val = mat_A.get_val()[j];

            row_norm[i] = std::max(row_norm[i], std::abs(val));
        }
    }

    for(int i = 0; i < mat_A.get_m(); i++)
    {
        if(std::abs(1.0 - row_norm[i]) > arg.tol)
        {
            return false;
        }
    }

    return true;
}
