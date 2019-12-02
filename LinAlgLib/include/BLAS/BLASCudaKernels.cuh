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

#ifndef __BLAS_CUDA_KERNELS_CUH__
#define __BLAS_CUDA_KERNELS_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <vector>
#include <random>	
#include <iostream>

// Level 1 BLAS

// apply givens rotation
template<typename T, int BlockSize>
__global__ void rot(T* d_x, T* d_y, T s, T c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	while (index + stride < n) {
		T x = d_x[index + stride];
		T y = d_y[index + stride];

		d_x[index + stride] = c * x + s * y;
		d_y[index + stride] = -s * x + c * y;

		stride += blockDim.x * gridDim.x;
	}
}

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

// sum of absolute values
template<typename T, int BlockSize>
__global__ void asum(const T* d_x, T* d_sum, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	T temp = 0.0;
	while (index + stride < n) {
		temp += fabs(d_x[index + stride]);

		stride += blockDim.x * gridDim.x;
	}

	smem[threadIdx.x] = temp;

	// parallel reduction;
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			smem[threadIdx.x] += smem[threadIdx.x + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_sum, smem[0]);
	}
}

// finds the (smallest) index of the element of the minimum abs value
template<typename T, int BlockSize>
__global__ void amin(const T* d_x, int* d_aminIndex, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	extern __shared__ int shared[];

	int indexOfThreadAMin = 0;
	T threadAMin = d_x[threadIdx.x];
	while (index + stride < n) {
		T temp = fabs(d_x[index + stride]);

		if (temp < threadAMin) {
			threadAMin = temp;
			indexOfThreadAMin = index + stride;
		}

		stride += blockDim.x * gridDim.x;
	}

	shared[threadIdx.x] = indexOfThreadAMin;

	// parallel reduction
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + i]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicMin(d_aminIndex, shared[0]);
	}
}

// finds the (smallest) index of the element of the maximum abs value
template<typename T, int BlockSize>
__global__ void amax(const T* d_x, int* d_amaxIndex, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0;

	extern __shared__ int shared[];

	int indexOfThreadAMax = 0;
	T threadAMax = d_x[threadIdx.x];
	while (index + stride < n) {
		T temp = fabs(d_x[index + stride]);

		if (temp < threadAMax) {
			threadAMax = temp;
			indexOfThreadAMax = index + stride;
		}

		stride += blockDim.x * gridDim.x;
	}

	shared[threadIdx.x] = indexOfThreadAMax;

	// parallel reduction
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + i]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicMax(d_amaxIndex, shared[0]);
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
