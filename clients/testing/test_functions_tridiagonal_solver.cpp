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

using namespace linalg;

bool Testing::test_tridiagonal_solver(Arguments arg)
{
    std::cout << "test_tridiagonal_solver" << std::endl;

    // Create a simple tridiagonal system for testing
    // System size
    int m = 8;
    int n = 1;

    std::cout << "n: " << n << std::endl;

    // Create tridiagonal matrix coefficients
    vector<float> lower_diag(m);
    vector<float> main_diag(m);
    vector<float> upper_diag(m);
    vector<float> rhs(m * n);
    vector<float> solution(m * n);

    // Initialize with a known system
    // Use a simple symmetric positive definite tridiagonal matrix
    // Main diagonal: 4.0
    // Off-diagonals: -1.0
    lower_diag[0]     = 0.0; // No lower diagonal for first row
    upper_diag[m - 1] = 0.0; // No upper diagonal
    for(int i = 0; i < m; i++)
    {
        main_diag[i] = 4.0;
        if(i > 0)
        {
            lower_diag[i - 1] = -1.0;
        }
        if(i < m - 1)
        {
            upper_diag[i] = -1.0;
        }
    }
    // for(size_t i = 0; i < m; i++)
    // {
    //     lower_diag[i] = i;
    //     main_diag[i]  = i + m;
    //     upper_diag[i] = i + 2 * m;
    // }

    // RHS set to make solution = 1.0 everywhere
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            rhs[m * i + j] = 1.0;
        }

        // Adjust boundary conditions
        rhs[m * i + 0]       = 3.0;
        rhs[m * i + (m - 1)] = 3.0;
    }

    // Move to device
    lower_diag.move_to_device();
    main_diag.move_to_device();
    upper_diag.move_to_device();
    rhs.move_to_device();
    solution.move_to_device();

    //for(int i = 0; i < 10; i++)
    //{
    tridiagonal_solver(m, n, lower_diag, main_diag, upper_diag, rhs, solution);
    //}
    linalg::sync();

    // Solve the system
    auto t1 = std::chrono::high_resolution_clock::now();
    //for(int i = 0; i < 100; i++)
    //{
    //    tridiagonal_solver(m, n, lower_diag, main_diag, upper_diag, rhs, solution);
    //}
    linalg::sync();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    // Move back to host for verification
    solution.move_to_host();
    main_diag.move_to_host();
    lower_diag.move_to_host();
    upper_diag.move_to_host();
    rhs.move_to_host();

    // Verify solution by computing residual: r = b - A*x
    vector<float> residual(m * n);
    float         max_residual = 0.0;

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            float ax = main_diag[j] * solution[m * i + j];
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

    //solution.print_vector("Solution");
    //residual.print_vector("Residual");

    std::cout << "Maximum residual: " << max_residual << std::endl;

    size_t total_bytes_read_write = sizeof(float) * (3 * m + 2 * m * n);
    float  total_gbytes           = (float)100 * total_bytes_read_write / 1e9;
    float  bandwidth              = total_gbytes / (ms_float.count() / 1e3);

    std::cout << "Total data transferred: " << total_gbytes << " GB"
              << " total_bytes_read_write: " << total_bytes_read_write << std::endl;

    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s"
              << " ms_float.count(): " << ms_float.count() << std::endl;

    // Check if solution is accurate enough
    float tolerance = 1e-6;
    bool  success   = (max_residual < tolerance);

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
