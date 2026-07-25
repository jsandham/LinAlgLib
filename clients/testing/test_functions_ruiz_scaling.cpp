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

bool testing::test_ruiz_scaling(Arguments arg)
{
    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    linalg::csr_matrix mat_A_copy;
    mat_A_copy.copy_from(mat_A);

    linalg::vector<double> D1(mat_A.get_m());
    linalg::vector<double> D2(mat_A.get_m());

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_device();
        mat_A_copy.move_to_device();
        D1.move_to_device();
        D2.move_to_device();
    }

    // Warmup
    for(int i = 0; i < 4; i++)
    {
        mat_A.apply_ruiz_scaling(D1, D2);
        mat_A.copy_from(mat_A_copy);
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        mat_A.apply_ruiz_scaling(D1, D2);
        mat_A.copy_from(mat_A_copy);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        mat_A.move_to_host();
        D1.move_to_host();
        D2.move_to_host();
    }

    return true;
}
