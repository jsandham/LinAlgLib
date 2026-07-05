//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

bool testing::test_exclusive_scan(Arguments arg)
{
    linalg::vector<double> init_vec(arg.m);
    linalg::vector<double> vec(arg.m);

    if(arg.backend == backend::GPU)
    {
        init_vec.move_to_device();
        vec.move_to_device();
    }

    init_vec.rand(-1.0, 1.0);

    // Warmup
    for(int i = 0; i < 4; i++)
    {
        vec.copy_from(init_vec);
        linalg::exclusive_scan(vec);
    }
    linalg::synchronize();

    // Timed run
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++)
    {
        vec.copy_from(init_vec);
        linalg::exclusive_scan(vec);
    }
    linalg::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    if(arg.backend == backend::GPU)
    {
        init_vec.move_to_host();
        vec.move_to_host();
    }

    // Verify results with a simple test case of all ones. The exclusive
    // scan of an array of ones should be [0, 1, 2, ..., n-1].
    bool   success        = true;
    double expected_value = 0.0;
    for(int i = 0; i < vec.get_size(); i++)
    {
        if(std::abs(vec[i] - expected_value) > 1e-9)
        {
            std::cout << "Mismatch at index " << i
                      << " std::abs(vec[i] - expected_value): " << std::abs(vec[i] - expected_value)
                      << std::endl;
            success = false;
        }

        expected_value += init_vec[i];
    }

    size_t total_bytes_read_write = sizeof(double) * 2 * arg.m;
    double total_gbytes           = (double)100 * total_bytes_read_write / 1e9;
    double bandwidth              = total_gbytes / (ms_float.count() / 1e3);
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return success;
}
