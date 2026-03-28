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

#include <cuda_runtime.h>

#include "cuda_scale.h"

#include "extract_diagonal_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// scale diagonal
//-------------------------------------------------------------------------------
void linalg::cuda_scale_diagonal(
    const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, double scalar)
{
    ROUTINE_TRACE("linalg::cuda_scale_diagonal_impl");
    scale_diagonal_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr, csr_col_ind, csr_val, scalar);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// scale by inverse diagonal
//-------------------------------------------------------------------------------
void linalg::cuda_scale_by_inverse_diagonal(
    const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, const double* diag)
{
    ROUTINE_TRACE("linalg::cuda_scale_by_inverse_diagonal_impl");
    scale_by_inverse_diagonal_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr, csr_col_ind, csr_val, diag);
    CHECK_CUDA_LAUNCH_ERROR();
}
