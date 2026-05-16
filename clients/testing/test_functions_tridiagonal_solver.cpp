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
#include <tuple>
#include <type_traits>
#include <utility>

#include "linalg.h"

using namespace linalg;

bool Testing::test_tridiagonal_solver(Arguments arg)
{
    // Create a simple tridiagonal system for testing
    // System size
    int m = arg.m;
    int n = arg.n;

    // Create tridiagonal matrix coefficients
    vector<double> lower_diag(m);
    vector<double> main_diag(m);
    vector<double> upper_diag(m);
    vector<double> rhs(m * n);
    vector<double> solution(m * n);

    // //1 3 0 0
    // //2 4 6 0
    // //0 5 7 9
    // //0 0 8 10
    // lower_diag[0] = 0.0f;
    // lower_diag[1] = 2.0f;
    // lower_diag[2] = 5.0f;
    // lower_diag[3] = 8.0f;
    // lower_diag[4] = 1.0f;
    // lower_diag[5] = 2.0f;
    // lower_diag[6] = 5.0f;
    // lower_diag[7] = 8.0f;

    // main_diag[0] = 1.0f;
    // main_diag[1] = 4.0f;
    // main_diag[2] = 7.0f;
    // main_diag[3] = 10.0f;
    // main_diag[4] = 1.0f;
    // main_diag[5] = 4.0f;
    // main_diag[6] = 7.0f;
    // main_diag[7] = 10.0f;

    // upper_diag[0] = 3.0f;
    // upper_diag[1] = 6.0f;
    // upper_diag[2] = 9.0f;
    // upper_diag[3] = 1.0f;
    // upper_diag[4] = 3.0f;
    // upper_diag[5] = 6.0f;
    // upper_diag[6] = 9.0f;
    // upper_diag[7] = 0.0f;

    // // 1 3 0 0 0 0 0 0
    // // 2 4 6 0 0 0 0 0
    // // 0 5 7 9 0 0 0 0
    // // 0 0 8 9 2 0 0 0
    // // 0 0 0 2 2 3 0 0
    // // 0 0 0 0 4 5 6 0
    // // 0 0 0 0 0 7 8 9
    // // 0 0 0 0 0 0 9 10
    // lower_diag[0] = 0.0f;
    // lower_diag[1] = 2.0f;
    // lower_diag[2] = 5.0f;
    // lower_diag[3] = 8.0f;
    // lower_diag[4] = 2.0f;
    // lower_diag[5] = 4.0f;
    // lower_diag[6] = 7.0f;
    // lower_diag[7] = 9.0f;
    // lower_diag[8] = 1.0f;
    // lower_diag[9] = 2.0f;
    // lower_diag[10] = 5.0f;
    // lower_diag[11] = 8.0f;
    // lower_diag[12] = 2.0f;
    // lower_diag[13] = 4.0f;
    // lower_diag[14] = 7.0f;
    // lower_diag[15] = 9.0f;

    // main_diag[0] = 1.0f;
    // main_diag[1] = 4.0f;
    // main_diag[2] = 7.0f;
    // main_diag[3] = 9.0f;
    // main_diag[4] = 2.0f;
    // main_diag[5] = 5.0f;
    // main_diag[6] = 8.0f;
    // main_diag[7] = 10.0f;
    // main_diag[8] = 1.0f;
    // main_diag[9] = 4.0f;
    // main_diag[10] = 7.0f;
    // main_diag[11] = 9.0f;
    // main_diag[12] = 2.0f;
    // main_diag[13] = 5.0f;
    // main_diag[14] = 8.0f;
    // main_diag[15] = 10.0f;

    // upper_diag[0] = 3.0f;
    // upper_diag[1] = 6.0f;
    // upper_diag[2] = 9.0f;
    // upper_diag[3] = 2.0f;
    // upper_diag[4] = 3.0f;
    // upper_diag[5] = 6.0f;
    // upper_diag[6] = 9.0f;
    // upper_diag[7] = 1.0f;
    // upper_diag[8] = 3.0f;
    // upper_diag[9] = 6.0f;
    // upper_diag[10] = 9.0f;
    // upper_diag[11] = 2.0f;
    // upper_diag[12] = 3.0f;
    // upper_diag[13] = 6.0f;
    // upper_diag[14] = 9.0f;
    // upper_diag[15] = 0.0f;

    // Initialize with a known system
    // Use a simple symmetric positive definite tridiagonal matrix
    // Main diagonal: 2.0
    // Off-diagonals: -0.5
    lower_diag[0]     = 0.0; // No lower diagonal for first row
    upper_diag[m - 1] = 0.0; // No upper diagonal
    for(int i = 0; i < m; i++)
    {
        // main_diag[i] = i % 8;
        main_diag[i] = 1.0;
        if(i > 0)
        {
            lower_diag[i] = 2.0;
        }
        if(i < m - 1)
        {
            upper_diag[i] = 2.0;
        }
    }

    // RHS set to make solution = 1.0 everywhere
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            rhs[m * i + j] = 1.0 + j;
        }

        // Adjust boundary conditions
        rhs[m * i + 0]       = 1.0;
        rhs[m * i + (m - 1)] = 1.0;
    }
    // for(int i = 0; i < n; i++)
    // {
    //     for(int j = 0; j < m; j++)
    //     {
    //         rhs[m * i + j] = m * i + j;
    //     }
    // }

    // Move to device
    lower_diag.move_to_device();
    main_diag.move_to_device();
    upper_diag.move_to_device();
    rhs.move_to_device();
    solution.move_to_device();

    tridiagonal_descr* descr = nullptr;
    create_tridiagonal_descr(&descr);
    // set_pivoting_strategy(descr, pivoting_strategy::none);
    set_pivoting_strategy(descr, pivoting_strategy::partial);

    tridiagonal_analysis(m, n, lower_diag, main_diag, upper_diag, descr);

    //for(int i = 0; i < 10; i++)
    //{
    tridiagonal_solver(m, n, lower_diag, main_diag, upper_diag, rhs, solution, descr);
    //}
    linalg::sync();

    // Solve the system
    auto t1 = std::chrono::high_resolution_clock::now();
    //for(int i = 0; i < 100; i++)
    //{
    //    tridiagonal_solver(m, n, lower_diag, main_diag, upper_diag, rhs, solution, descr);
    //}
    //linalg::sync();
    auto t2 = std::chrono::high_resolution_clock::now();

    destroy_tridiagonal_descr(descr);

    std::chrono::duration<double, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    // Move back to host for verification
    solution.move_to_host();
    main_diag.move_to_host();
    lower_diag.move_to_host();
    upper_diag.move_to_host();
    rhs.move_to_host();

    // Verify solution by computing residual: r = b - A*x
    vector<double> residual(m * n);
    double         max_residual = 0.0;

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            double ax = main_diag[j] * solution[m * i + j];
            if(j > 0)
            {
                ax += lower_diag[j] * solution[m * i + j - 1];
            }
            if(j < m - 1)
            {
                ax += upper_diag[j] * solution[m * i + j + 1];
            }
            residual[m * i + j] = std::abs(rhs[m * i + j] - ax);
            max_residual        = std::max(max_residual, residual[m * i + j]);
        }
    }

    solution.print_vector("Solution");
    residual.print_vector("Residual");

    std::cout << "Maximum residual: " << max_residual << std::endl;

    size_t total_bytes_read_write = sizeof(double) * (3 * m + 2 * m * n);
    double total_gbytes           = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth              = total_gbytes / (ms_float.count() / 1e3);

    std::cout << "Total data transferred: " << total_gbytes << " GB"
              << " total_bytes_read_write: " << total_bytes_read_write << std::endl;

    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s"
              << " ms_float.count(): " << ms_float.count() << std::endl;

    // Check if solution is accurate enough
    double tolerance = 1e-6;
    bool   success   = (max_residual < tolerance);

    if(!success)
    {
        std::cout << "Test FAILED: residual exceeds tolerance" << std::endl;
    }
    else
    {
        std::cout << "Test PASSED" << std::endl;
    }

    return success;
}
