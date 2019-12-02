//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
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

#ifndef __SPARSE_BLAS_CUDA_KERNELS_CUH__
#define __SPARSE_BLAS_CUDA_KERNELS_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <random>	
#include <iostream>

// Level 2 sparse BLAS

// y = alpha * A * x + beta * y
template<typename T, int BlockSize>
__global__ void crs_gemv(const int* d_rowPtr, const int* d_col, const T* d_val, const T* d_x, T* d_y, T alpha, T beta, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		T  temp = 0.0;
		for (int i = d_rowPtr[index + stride]; i < d_rowPtr[index + stride + 1]; i++) {
			temp += d_val[i] * d_x[d_col[i]];
		}

		d_y[index + stride] += alpha * temp + beta * d_y[index + stride];

		stride += blockDim.x * gridDim.x;
	}
}

// y = alpha * A * x + beta * y where A is symmetric
template<typename T, int BlockSize>
__global__ void crs_symv(const int* d_rowPtr, const int* d_col, const T* d_val, const T* d_x, T* d_y, T alpha, T beta, int n)
{

}

// Level 3 sparse BLAS

// C = alpha * A * B + beta * C
template<typename T, int BlockSize>
__global__ void csr_gemm()
{

}

#endif
