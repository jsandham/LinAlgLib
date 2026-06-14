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

#include <cmath>
#include <cuda_runtime.h>

#include "cuda_math.h"
#include "cuda_primitives.h"

#include "preconditioner_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// infinity norm
//-------------------------------------------------------------------------------
template <typename T>
T linalg::cuda_norm_inf(const T* array, int size)
{
    ROUTINE_TRACE("linalg::cuda_norm_inf_impl");
    return cuda_find_maximum(size, array);
}

//-------------------------------------------------------------------------------
// jacobi solve
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_jacobi_solve(const T* rhs, const T* diag, T* x, size_t size)
{
    ROUTINE_TRACE("linalg::cuda_jacobi_solve_impl");
    jacobi_solve_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, rhs, diag, x);
    CHECK_CUDA_LAUNCH_ERROR();
}

template double linalg::cuda_norm_inf<double>(const double*, int);
template float  linalg::cuda_norm_inf<float>(const float*, int);
template void   linalg::cuda_jacobi_solve<double>(const double*, const double*, double*, size_t);
template void   linalg::cuda_jacobi_solve<float>(const float*, const float*, float*, size_t);
