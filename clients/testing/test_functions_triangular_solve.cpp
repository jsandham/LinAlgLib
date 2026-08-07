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

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "linalg.h"

bool testing::test_triangular_solve(Arguments arg)
{
    std::cout << "Testing triangular solve with matrix: " << arg.filename << std::endl;

    linalg::csr_matrix<double> mat_A;
    mat_A.read_mtx(arg.filename);

    assert(mat_A.get_m() == mat_A.get_n());

    linalg::csr_matrix<double> mat_B;

    switch(arg.uplo)
    {
    case testing::uplo::lower:
        mat_B.copy_lower_triangular_from(mat_A, true);
        break;
    case testing::uplo::upper:
        mat_B.copy_upper_triangular_from(mat_A, true);
        break;
    }

    linalg::vector<double> vec_z(mat_A.get_m());
    linalg::vector<double> vec_y(mat_A.get_m());
    linalg::vector<double> vec_x(mat_A.get_m());

    if(arg.backend == backend::GPU)
    {
        mat_B.move_to_device();
        vec_z.move_to_device();
        vec_y.move_to_device();
        vec_x.move_to_device();
    }

    vec_z.ones();
    vec_y.zeros();
    vec_x.zeros();

    // y = A * z
    mat_B.multiply_by_vector(vec_y, vec_z);

    // Warmup (solve A * x = y)
    for(int i = 0; i < 4; i++)
    {
        switch(arg.uplo)
        {
        case testing::uplo::lower:
            mat_B.triangular_solve_lower(vec_x, vec_y, false);
            break;
        case testing::uplo::upper:
            mat_B.triangular_solve_upper(vec_x, vec_y, false);
            break;
        }
    }
    linalg::synchronize();

    // Timed run (solve A * x = y)
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        switch(arg.uplo)
        {
        case testing::uplo::lower:
            mat_B.triangular_solve_lower(vec_x, vec_y, false);
            break;
        case testing::uplo::upper:
            mat_B.triangular_solve_upper(vec_x, vec_y, false);
            break;
        }
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_B.move_to_host();
        vec_x.move_to_host();
        vec_y.move_to_host();
        vec_z.move_to_host();
    }

    // Verify solution. Since we solved A * x = y, we can check if x is equal to z.
    bool success = true;
    for(int i = 0; i < vec_x.get_size(); i++)
    {
        if(std::abs(vec_x[i] - vec_z[i]) > 1e-6)
        {
            std::cout << "Mismatch at index " << i << ": x = " << vec_x[i] << ", z = " << vec_z[i]
                      << std::endl;
            success = false;
        }
    }

    return success;
}
