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

using namespace linalg;

bool Testing::test_transpose(Arguments arg)
{
    csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    csr_matrix mat_A_transpose;
    csr_matrix mat_A2;

    mat_A_transpose.resize(mat_A.get_n(), mat_A.get_m(), mat_A.get_nnz());
    mat_A2.resize(mat_A.get_m(), mat_A.get_n(), mat_A.get_nnz());

    mat_A.move_to_device();
    mat_A_transpose.move_to_device();
    mat_A2.move_to_device();

    auto t1 = std::chrono::high_resolution_clock::now();
    mat_A.transpose(mat_A_transpose);
    mat_A_transpose.transpose(mat_A2);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << std::endl;

    mat_A.move_to_host();
    mat_A_transpose.move_to_host();
    mat_A2.move_to_host();

    return check_matrix_equality(mat_A, mat_A2);
}