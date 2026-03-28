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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
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

#include "device_math.h"

#include <cmath>
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_math.h"
#endif

double linalg::device_norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device_norm_euclid");

    return std::sqrt(device_dot_product(array, array));
}

double linalg::device_norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device_norm_inf");

    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_norm_inf(array.get_vec(), array.get_size()));
    }

    std::cout << "Error: Not device backend available for the function " << __func__ << std::endl;
    return 0.0;
}

void linalg::device_jacobi_solve(const vector<double>& rhs,
                                 const vector<double>& diag,
                                 vector<double>&       x)
{
    ROUTINE_TRACE("linalg::device_jacobi_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_jacobi_solve(rhs.get_vec(), diag.get_vec(), x.get_vec(), rhs.get_size()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
