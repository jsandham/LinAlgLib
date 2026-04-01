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

using namespace linalg;

bool Testing::test_exclusive_scan(Arguments arg)
{
    // Host solution
    vector<double> vec_1(arg.m);
    vec_1.ones();

    exclusive_scan(vec_1);

    // Device solution
    vector<double> vec_2(arg.m);
    vec_2.ones();

    vec_2.move_to_device();

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 4; i++)
    {
        vec_2.ones();
        exclusive_scan(vec_2);
    }
    linalg::sync();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> ms_float = t2 - t1;
    std::cout << "Solve time: " << ms_float.count() << "ms" << std::endl;

    vec_2.move_to_host();

    //vec_1.print_vector("Host solution");
    //vec_2.print_vector("Device solution");

    bool pass = true;
    for(int i = 0; i < vec_2.get_size(); i++)
    {
        if(vec_2[i] != vec_1[i])
        {
            std::cout << "Mismatch at index " << i << ": " << vec_2[i] << " != " << vec_1[i]
                      << std::endl;
            pass = false;
            break;
        }
    }

    return pass;
}
