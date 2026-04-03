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

#include "device_tridiagonal.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_tridiagonal.h"
#endif

void linalg::free_tridiagonal_device_data(tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::free_tridiagonal_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_tridiagonal_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_tridiagonal_analysis(int                  m,
                                         int                  n,
                                         const vector<float>& lower_diag,
                                         const vector<float>& main_diag,
                                         const vector<float>& upper_diag,
                                         tridiagonal_descr*   descr)
{
    ROUTINE_TRACE("linalg::device_tridiagonal_analysis");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_tridiagonal_analysis(
            m, n, lower_diag.get_vec(), main_diag.get_vec(), upper_diag.get_vec(), descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_tridiagonal_solver(int                      m,
                                       int                      n,
                                       const vector<float>&     lower_diag,
                                       const vector<float>&     main_diag,
                                       const vector<float>&     upper_diag,
                                       const vector<float>&     b,
                                       vector<float>&           x,
                                       const tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::device_tridiagonal_solver");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_tridiagonal_solver(m,
                                          n,
                                          lower_diag.get_vec(),
                                          main_diag.get_vec(),
                                          upper_diag.get_vec(),
                                          b.get_vec(),
                                          x.get_vec(),
                                          descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
