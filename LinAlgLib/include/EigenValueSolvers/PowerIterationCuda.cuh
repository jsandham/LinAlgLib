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

#ifndef __POWER_ITERATION_CUDA_CUH__
#define __POWER_ITERATION_CUDA_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <random>	
#include <iostream>

template<typename T, int NumThreadPerBlock>
__global__ void updateEigenValue(const int* d_rowPtr, const int* d_col, const T* d_val, const T* d_b, T* d_Ab, T* d_normAbSqr, T* d_bAb, T* d_bb, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	// calculate A*b
	while (index + stride < n) {
		d_Ab[index + stride] = 0.0f;
		for (int i = d_rowPtr[index + stride]; i < d_rowPtr[index + stride + 1]; i++) {
			d_Ab[index + stride] += d_val[i] * d_b[d_col[i]];
		}

		stride += blockDim.x * gridDim.x;
	}

	//calculate dot(temp, temp), dot(b, temp), and dot(b, b)
	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	T* cache1 = &smem[0];
	T* cache2 = &smem[NumThreadPerBlock];
	T* cache3 = &smem[2* NumThreadPerBlock];

	stride = 0;
	T t1 = 0.0f;
	T t2 = 0.0f;
	T t3 = 0.0f;
	while (index + stride < n) {
		t1 += d_Ab[index + stride] * d_Ab[index + stride];
		t2 += d_b[index + stride] * d_Ab[index + stride];
		t3 += d_b[index + stride] * d_b[index + stride];
		stride += blockDim.x * gridDim.x;
	}

	cache1[threadIdx.x] = t1;
	cache2[threadIdx.x] = t2;
	cache3[threadIdx.x] = t3;

	// perform reduction on shared cache to get final dot products
	/*int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache1[threadIdx.x] += cache1[threadIdx.x + i];
			cache2[threadIdx.x] += cache2[threadIdx.x + i];
			cache3[threadIdx.x] += cache3[threadIdx.x + i];
		}

		__syncthreads();
		i /= 2;
	}*/
	for (unsigned int i = blockDim.x / 2; i > 0; i /= 2/* i >>= 1*/) {
		if (threadIdx.x < i) {
			cache1[threadIdx.x] += cache1[threadIdx.x + i];
			cache2[threadIdx.x] += cache2[threadIdx.x + i];
			cache3[threadIdx.x] += cache3[threadIdx.x + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_normAbSqr, cache1[0]);
		atomicAdd(d_bAb, cache2[0]);
		atomicAdd(d_bb, cache3[0]);
	}
}

template<typename T>
__global__ void updateEigenVector(T* d_b, const T* d_Ab, const T* d_normAbSqr, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	T normAb = sqrt(*d_normAbSqr);

	while (index + stride < n) {
		d_b[index] = d_Ab[index] / normAb;

		stride += blockDim.x * gridDim.x;
	}
}

template<typename T>
__global__ void updateResidualSqr(const T* d_b, const T* d_Ab, T lambda, T* d_resSqr, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	extern __shared__ __align__(sizeof(T)) unsigned char shared[];
	T* smem = reinterpret_cast<T*>(shared);

	T temp = 0.0f;
	while (index + stride < n) {
		T t = d_Ab[index + stride] - lambda * d_b[index + stride];
		temp += t * t;

		stride += blockDim.x * gridDim.x;
	}

	smem[threadIdx.x] = temp;

	// parallel reduction
	/*int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			smem[threadIdx.x] += smem[threadIdx.x + i];

			__syncthreads();
		}

		i /= 2;
	}*/
	for (unsigned int i = blockDim.x / 2; i > 0; i /= 2/* i >>= 1*/) {
		if(threadIdx.x < i) {
			smem[threadIdx.x] += smem[threadIdx.x + i];
		}

		__syncthreads();
	}




	if (threadIdx.x == 0) {
		atomicAdd(d_resSqr, smem[0]);
	}
}

template<typename T, int NumThreadsPerBlock> // rename NumThreadsPerBlock to BlockSize?
T powerIterationCuda(const int* rowptr, const int* col, const T* val, T* eigenVec, T tol, int n, int maxIter)
{
	// host
	T h_normAbSqr = 0.0f;
	T h_bAb = 0.0f;
	T h_bb = 0.0f;
	T h_resSqr = 0.0;
	T lambda = 0.0f;

	std::vector<T> h_b; // intial eigenvector guess
	std::vector<T> h_Ab;
	h_b.resize(n);
	h_Ab.resize(n);

	std::default_random_engine generator;
	std::uniform_real_distribution<T> distribution(0.0, 1.0);

	for (int i = 0; i < n; i++) {
		T number = distribution(generator);
		h_b[i] = 1.0f;// 2.0 * number - 1.0;
		h_Ab[i] = 0.0;
	}

	// device
	int* d_rowPtr;
	int* d_col;
	T* d_val;
	T* d_b;
	T* d_Ab;
	T* d_normAbSqr;
	T* d_bAb;
	T* d_bb;
	T* d_resSqr;

	int nnz = rowptr[n];

	cudaMalloc((void**)&d_rowPtr, (n + 1) * sizeof(int));
	cudaMalloc((void**)&d_col, nnz * sizeof(int));
	cudaMalloc((void**)&d_val, nnz * sizeof(T));
	cudaMalloc((void**)&d_b, n * sizeof(T));
	cudaMalloc((void**)&d_Ab, n * sizeof(T));
	cudaMalloc((void**)&d_normAbSqr, sizeof(T));
	cudaMalloc((void**)&d_bAb, sizeof(T));
	cudaMalloc((void**)&d_bb, sizeof(T));
	cudaMalloc((void**)&d_resSqr, sizeof(T));

	cudaMemcpy(d_rowPtr, rowptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nnz * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b[0], n * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ab, &h_Ab[0], n * sizeof(T), cudaMemcpyHostToDevice);

	unsigned int numOfBlocks = (n - 1) / NumThreadsPerBlock + 1;

	dim3 gridSize(numOfBlocks, 1);
	dim3 blockSize(NumThreadsPerBlock, 1);
	size_t smemSize = NumThreadsPerBlock * sizeof(T);
	for (int k = 0; k < maxIter; k++) {
		updateEigenValue<T, NumThreadsPerBlock> << <gridSize, blockSize, 3 * smemSize >> > (d_rowPtr, d_col, d_val, d_b, d_Ab, d_normAbSqr, d_bAb, d_bb, n);

		cudaMemcpy(&h_normAbSqr, d_normAbSqr, sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_bAb, d_bAb, sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_bb, d_bb, sizeof(T), cudaMemcpyDeviceToHost);

		lambda = h_bAb / h_bb;

		updateEigenVector<T> << <gridSize, blockSize, smemSize >> > (d_b, d_Ab, d_normAbSqr, n);

		updateResidualSqr<T> << <gridSize, blockSize, smemSize >> > (d_b, d_Ab, lambda, d_resSqr, n);

		cudaMemcpy(&h_resSqr, d_resSqr, sizeof(T), cudaMemcpyDeviceToHost);

		std::cout << "eigen value: " << lambda << " normABSqr: " << h_normAbSqr << " bAb: " << h_bAb << " bb: " << h_bb << " resSqr: " << h_resSqr << std::endl;

		if (h_resSqr < tol) {
			break;
		}

		h_normAbSqr = 0.0f;
		h_bAb = 0.0f;
		h_bb = 0.0f;
		h_resSqr = 0.0f;

		cudaMemcpy(d_normAbSqr, &h_normAbSqr, sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bAb, &h_bAb, sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bb, &h_bb, sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_resSqr, &h_resSqr, sizeof(T), cudaMemcpyHostToDevice);
	}

	// final eigen value
	lambda = h_bAb / h_bb;

	// final eigen vector
	for (int i = 0; i < n; i++) {
		eigenVec[i] = h_b[i];
	}

	cudaFree(d_rowPtr);
	cudaFree(d_col);
	cudaFree(d_val);
	cudaFree(d_b);
	cudaFree(d_Ab);
	cudaFree(d_normAbSqr);
	cudaFree(d_bAb);
	cudaFree(d_bb);
	cudaFree(d_resSqr);

	return lambda;
}

#endif