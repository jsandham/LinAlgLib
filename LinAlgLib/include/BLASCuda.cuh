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

#ifndef __BLAS_CUDA_CUH__
#define __BLAS_CUDA_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <random>	
#include <iostream>

// Level 1 BLAS

// swap x and y
template<typename T, int BlockSize>
__global__ void swap(T* d_x, T* d_y, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		T temp = d_y[index + stride];
		d_y[index + stride] = d_x[index + stride];
		d_x[index + stride] = temp;

		stride += blockDim.x * gridDim.x;
	}
}

// x = alpha * x;
template<typename T, int BlockSize>
__global__ void scal(T* d_x, T alpha, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		d_x[index + stride] += alpha * d_x[index + stride];

		stride += blockDim.x * gridDim.x;
	}
}

// y = x
template<typename T, int BlockSize>
__global__ void copy(const T* d_c, T* d_y, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		d_y[index + stride] = d_x[index + stride];

		stride += blockDim.x * gridDim.x;
	}
}

// y = alpha * x + y
template<typename T, int BlockSize>
__global__ void axpy(const T* d_x, T* d_y, T alpha, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		d_y[index + stride] += alpha * d_x[index + stride];

		stride += blockDim.x * gridDim.x;
	}
}

// dot product 
template<typename T, int BlockSize>
__global__ void dot(const T* d_x, const T* d_y, T* d_product, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	T temp = 0.0;
	while (index + stride < n) {
		temp += d_x[index + stride] * d_y[index + stride];

		stride += blockDim.x * gridDim.x;
	}

	smem[threadIdx.x] = temp;

	// parallel reduction
	for (unsigned int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			smem[threadIdx.x] += smem[threadIdx.x + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_product, smem[0]);
	}
}


// Level 2 BLAS

// y = alpha * A * x + beta * y
template<typename T, int BlockSize>
__global__ void gemv(const T* d_A, const T* d_x, T* d_y, T alpha, T beta, int nrow, int ncol)
{
	int index = threadIdx.x + blockIdx.x * bloxkDim.x;

	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	T temp = 0.0;
	for (int i = 0; i < (ncol + blockDim.x - 1) / blockDim.x; i++) {
		if (threadIdx.x + i * blockDim.x < ncol) {
			smem[threadIdx.x] = d_x[threadIdx.x + blockDim.x * i];
		}
		else {
			smem[threadIdx.x] = 0.0;
		}

		__syncthreads();

		if (index < nrow) {
			for (int j = 0; j < blockDim.x; j++) {
				temp += d_A[index * ncol + (i * blockDim.x + j)] * smem[j]; // assuming A is row major ordering
			}
		}
	}

	if (index < nrow) {
		d_y[index] += alpha * temp + beta * d_y[index];
	}
}

// Level 3 BLAS

// C = alpha * A * B + beta * C
template<typename T, int BlockSize>
__global__ void gemm(const T* d_A, const T* d_B, T* d_C, T alpha, T beta, int nrow, int ncol)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	for (int k = 0; k < ncol; k++) {
		T temp = 0.0;
		for (int i = 0; i < (ncol + blockDim.x - 1) / blockDim.x; i++) {
			if (threadIdx.x + i * blockDim.x < ncol) {
				smem[threadIdx.x] = d_B[(threadIdx.x + blockDim.x * i) * ncol + k * nrow]; // assuming B is row major ordering
			}
			else {
				smem[threadIdx.x] = 0.0;
			}

			__syncthreads();

			if (index < nrow) {
				for (int j = 0; j < blockDim.x; j++) {
					temp += d_A[index * ncol + (i * blockDim.x + j)] * smem[j]; // assuming A is row major ordering
				}
			}
		}

		if (index < nrow) {
			d_y[index] += alpha * temp + beta * d_y[index];
		}
	}
}

#endif
