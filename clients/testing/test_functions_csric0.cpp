//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

using namespace linalg;

bool Testing::test_csric0(Arguments arg)
{
    csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    std::cout << "Matrix: " << arg.filename << " m: " << mat_A.get_m() << " n: " << mat_A.get_n()
              << " nnz: " << mat_A.get_nnz() << std::endl;

    // Make a copy for host computation
    csr_matrix mat_A_host;
    mat_A_host.copy_from(mat_A);

    // Make a copy for device computation
    csr_matrix mat_A_device;
    mat_A_device.copy_from(mat_A);

    // Prepare for csric0 analysis
    csric0_descr* descr = nullptr;
    create_csric0_descr(&descr);

    int structural_zero = 0;
    int numeric_zero    = 0;

    // Perform analysis on host
    csric0_analysis(mat_A_host, descr);

    // Compute csric0 on the host
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++)
    {
        csric0_compute(mat_A_host, descr);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_host = t2 - t1;
    std::cout << "host csric0: " << ms_host.count() << "ms" << std::endl;

    // Move to device
    mat_A_device.move_to_device();

    // Perform analysis on device
    csric0_analysis(mat_A_device, descr);

    // Compute csric0 on the device
    auto t3 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++)
    {
        csric0_compute(mat_A_device, descr);
    }
    linalg::sync();
    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_device = t4 - t3;
    std::cout << "device csric0: " << ms_device.count() << "ms" << std::endl;

    // Move device result to host for comparison
    mat_A_device.move_to_host();

    mat_A_device.print_matrix("A");

    // Compare host and device solution
    double max_error = 0.0;
    for(int i = 0; i < mat_A_host.get_nnz(); i++)
    {
        double host_val   = mat_A_host.get_val()[i];
        double device_val = mat_A_device.get_val()[i];
        max_error         = std::max(max_error, std::abs(device_val - host_val));
    }

    std::cout << "max_error: " << max_error << std::endl;

    destroy_csric0_descr(descr);

    if(max_error > 1e-12)
    {
        return false;
    }

    return true;
}
