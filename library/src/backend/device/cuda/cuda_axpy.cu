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

#include "cuda_axpy.h"

#include "axpby_kernels.cuh"
#include "dot_product_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// Compute y = alpha * x + y
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_axpy(int size, T alpha, const T* x, T* y)
{
    ROUTINE_TRACE("linalg::cuda_axpy_impl");
    axpy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute y = alpha * x + beta * y
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_axpby(int size, T alpha, const T* x, T beta, T* y)
{
    ROUTINE_TRACE("linalg::cuda_axpby_impl");
    axpby_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute z = alpha * x + beta * y + gamma * z
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_axpbypgz(int size, T alpha, const T* x, T beta, const T* y, T gamma, T* z)
{
    ROUTINE_TRACE("linalg::cuda_axpbypgz_impl");
    axpbypgz_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y, gamma, z);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
template <typename T>
T linalg::cuda_dot_product(const T* x, const T* y, int size)
{
    ROUTINE_TRACE("linalg::cuda_dot_product_impl");
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    dot_product_kernel_part1<256><<<256, 256>>>(size, x, y, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    dot_product_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    T result;
    CHECK_CUDA(cudaMemcpy(&result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));

    return result;
}

template void linalg::cuda_axpy<double>(int, double, const double*, double*);
template void linalg::cuda_axpy<float>(int, float, const float*, float*);
template void linalg::cuda_axpby<double>(int, double, const double*, double, double*);
template void linalg::cuda_axpby<float>(int, float, const float*, float, float*);
template void linalg::cuda_axpbypgz<double>(
    int, double, const double*, double, const double*, double, double*);
template void
    linalg::cuda_axpbypgz<float>(int, float, const float*, float, const float*, float, float*);
template double linalg::cuda_dot_product<double>(const double*, const double*, int);
template float  linalg::cuda_dot_product<float>(const float*, const float*, int);
