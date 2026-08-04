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

bool testing::test_dot_product(Arguments arg)
{
    const size_t size = arg.m;

    linalg::vector<double> x(size);
    linalg::vector<double> y(size);
    x.fill(2.0);
    y.fill(3.0);

    if(arg.backend == backend::GPU)
    {
        x.move_to_device();
        y.move_to_device();
    }

    double result = 0.0;

    // Warmup
    for(int i = 0; i < 4; i++)
    {
        result = linalg::dot_product(x, y);
    }
    linalg::synchronize();

    // Timed solve
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        result = linalg::dot_product(x, y);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        x.move_to_host();
        y.move_to_host();
    }

    std::cout << "Dot product result: " << result << std::endl;

    bool success = (size * 2.0 * 3.0 == result);

    size_t total_bytes_read       = sizeof(double) * 2 * size;
    size_t total_bytes_written    = sizeof(double);
    size_t total_bytes_read_write = total_bytes_read + total_bytes_written;
    double total_gbytes           = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth              = total_gbytes / (ms_float.count() / 1e3);

    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return success;
}
