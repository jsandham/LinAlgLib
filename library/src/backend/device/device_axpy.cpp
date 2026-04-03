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

#include "device_axpy.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_axpy.h"
#endif

void linalg::device_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpy");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpy(x.get_size(), alpha, x.get_vec(), y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpby");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpby(x.get_size(), alpha, x.get_vec(), beta, y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_axpbypgz(double                alpha,
                             const vector<double>& x,
                             double                beta,
                             const vector<double>& y,
                             double                gamma,
                             vector<double>&       z)
{
    ROUTINE_TRACE("linalg::device_axpbypgz");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_axpbypgz(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

double linalg::device_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_dot_product");
    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_dot_product(x.get_vec(), y.get_vec(), x.get_size()));
    }
    std::cout << "Error: Not device backend available for the function " << __func__ << std::endl;
    return 0.0;
}
