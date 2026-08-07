//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024-2026 James Sandham
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

bool testing::test_sptrsv(Arguments arg)
{
    linalg::csr_matrix<double> mat_A;
    mat_A.read_mtx(arg.filename);

    linalg::vector<double> vec_x(mat_A.get_n());
    vec_x.ones();

    linalg::vector<double> vec_y1(mat_A.get_m());
    vec_y1.ones();

    linalg::vector<double> vec_y2(mat_A.get_m());
    vec_y2.copy_from(vec_y1);

    // Prepare for csrtrsv analysis
    linalg::csrtrsv_descr* descr = nullptr;
    linalg::create_csrtrsv_descr(&descr);

    //linalg::csrtrsv_analysis(mat_A, linalg::triangular_type::lower, linalg::diagonal_type::non_unit, descr);
    linalg::csrtrsv_analysis(
        mat_A, linalg::triangular_type::upper, linalg::diagonal_type::non_unit, descr);

    // Multiple by vector on the host
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 4; i++)
    {
        //linalg::csrtrsv_solve(
        //    mat_A, vec_x, vec_y1, 1.0, linalg::triangular_type::lower, linalg::diagonal_type::non_unit, descr);
        linalg::csrtrsv_solve(mat_A,
                              vec_x,
                              vec_y1,
                              1.0,
                              linalg::triangular_type::upper,
                              linalg::diagonal_type::non_unit,
                              descr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_host = t2 - t1;
    std::cout << "host sptrsv: " << ms_host.count() << "ms" << std::endl;

    // vec_y1.print_vector("vec_y1");

    mat_A.move_to_device();
    vec_x.move_to_device();
    vec_y2.move_to_device();

    //linalg::csrtrsv_analysis(mat_A, linalg::triangular_type::lower, linalg::diagonal_type::non_unit, descr);
    linalg::csrtrsv_analysis(
        mat_A, linalg::triangular_type::upper, linalg::diagonal_type::non_unit, descr);

    // Multiple by vector on the device
    auto t3 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++)
    {
        //linalg::csrtrsv_solve(
        //    mat_A, vec_x, vec_y2, 1.0, linalg::triangular_type::lower, linalg::diagonal_type::non_unit, descr);
        linalg::csrtrsv_solve(mat_A,
                              vec_x,
                              vec_y2,
                              1.0,
                              linalg::triangular_type::upper,
                              linalg::diagonal_type::non_unit,
                              descr);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_device = t4 - t3;
    std::cout << "device sptrsv: " << ms_device.count() << "ms" << std::endl;

    vec_y2.move_to_host();
    // vec_y2.print_vector("vec_y2");

    // Compare host and device solution
    double max_error = 0.0;
    for(int i = 0; i < mat_A.get_m(); i++)
    {
        max_error = std::max(max_error, std::abs(vec_y2[i] - vec_y1[i]));
        if(std::abs(vec_y2[i] - vec_y1[i]) > 1e-12)
        {
            std::cout << "vec_y1[i]: " << vec_y1[i] << " vec_y2[i]: " << vec_y2[i]
                      << " std::abs(vec_y2[i] - vec_y1[i]): " << std::abs(vec_y2[i] - vec_y1[i])
                      << " i: " << i << std::endl;
            break;
        }
    }

    std::cout << "max_error: " << max_error << std::endl;

    linalg::destroy_csrtrsv_descr(descr);

    if(max_error > 1e-12)
    {
        return false;
    }

    return true;
}
