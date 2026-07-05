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

bool testing::test_tridiagonal_solver(Arguments arg)
{
    // System size
    int m = arg.m;
    int n = arg.n;

    // Create tridiagonal matrix coefficients
    linalg::vector<double> lower_diag(m);
    linalg::vector<double> main_diag(m);
    linalg::vector<double> upper_diag(m);
    linalg::vector<double> rhs(m * n);
    linalg::vector<double> solution(m * n);

    std::mt19937 gen(123456);

    // 3. Define the range [0.0, 1.0)
    std::uniform_real_distribution<double> main_dist(2.0, 2.5);
    std::uniform_real_distribution<double> lower_dist(0.5, 1.0);
    std::uniform_real_distribution<double> upper_dist(0.5, 1.0);

    // Initialize with the same system as test_tridiagonal_solver
    lower_diag[0]     = 0.0;
    upper_diag[m - 1] = 0.0;
    for(int i = 0; i < m; i++)
    {
        main_diag[i] = main_dist(gen);
        if(i > 0)
        {
            lower_diag[i] = lower_dist(gen);
        }
        if(i < m - 1)
        {
            upper_diag[i] = upper_dist(gen);
        }
    }

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            rhs[m * i + j] = 1.0;
        }
        rhs[m * i + 0]       = 1.0;
        rhs[m * i + (m - 1)] = 1.0;
    }

    // Create the solver
    linalg::pivoting_strategy pivoting;
    switch(arg.pivoting_strategy)
    {
    case testing::pivoting_strategy::None:
        pivoting = linalg::pivoting_strategy::none;
        break;
    case testing::pivoting_strategy::Partial:
        pivoting = linalg::pivoting_strategy::partial;
        break;
    }
    linalg::tridiagonal_solver solver(m, n, pivoting);

    if(arg.backend == backend::GPU)
    {
        // To run on device: move data and solver workspace to device
        lower_diag.move_to_device();
        main_diag.move_to_device();
        upper_diag.move_to_device();
        rhs.move_to_device();
        solution.move_to_device();
        solver.move_to_device();
    }

    // Warmup
    for(int i = 0; i < 10; i++)
    {
        solver.solve(lower_diag, main_diag, upper_diag, rhs, solution);
    }
    linalg::synchronize();

    // Timed solve
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        solver.solve(lower_diag, main_diag, upper_diag, rhs, solution);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        // Move back to host for verification (no-op if already on host)
        solution.move_to_host();
        lower_diag.move_to_host();
        main_diag.move_to_host();
        upper_diag.move_to_host();
        rhs.move_to_host();
        solver.move_to_host();
    }

    // Verify solution by computing residual: r = b - A*x
    double max_residual = 0.0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            double ax = main_diag[j] * solution[m * i + j];
            if(j > 0)
            {
                ax = std::fma(lower_diag[j], solution[m * i + j - 1], ax);
            }
            if(j < m - 1)
            {
                ax = std::fma(upper_diag[j], solution[m * i + j + 1], ax);
            }
            max_residual = std::max(max_residual, std::abs(rhs[m * i + j] - ax));
        }
    }

    std::cout << "Maximum residual: " << max_residual << std::endl;

    size_t total_bytes_read_write = sizeof(double) * (3 * m + 2 * m * n);
    double total_gbytes           = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth              = total_gbytes / (ms_float.count() / 1e3);

    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    double tolerance = 1e-15;
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
