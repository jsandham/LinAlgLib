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

#include "device_scale.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_scale.h"
#endif

void linalg::device_scale_diagonal(csr_matrix& A, double scalar)
{
    ROUTINE_TRACE("linalg::device_scale_diagonal");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_scale_diagonal(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), scalar));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag)
{
    ROUTINE_TRACE("linalg::device_scale_by_inverse_diagonal");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_scale_by_inverse_diagonal(
            A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), diag.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
