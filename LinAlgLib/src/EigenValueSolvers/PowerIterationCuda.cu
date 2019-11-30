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

#include "../include/EigenValueSolvers/PowerIterationCuda.cuh"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <random>	
#include <iostream>

float powerIterationCuda(const int* rowptr, const int* col, const float* val, float* eigenVec, float tol, int n, int maxIter)
{
	// host
	float h_normAbSqr = 0.0f;
	float h_bAb = 0.0f;
	float h_bb = 0.0f;
	float h_resSqr = 0.0;
	float lambda = 0.0f;

	std::vector<float> h_b; // intial eigenvector guess
	std::vector<float> h_Ab; 
	h_b.resize(n);
	h_Ab.resize(n);

	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0, 1.0);

	for (int i = 0; i < n; i++) {
		float number = distribution(generator);
		h_b[i] = 1.0f;// 2.0 * number - 1.0;
		h_Ab[i] = 0.0;
	}

	// device
	int* d_rowPtr;
	int* d_col;
	float* d_val;
	float* d_b;
	float* d_Ab;
	float* d_normAbSqr;
	float* d_bAb;
	float* d_bb;
	float* d_resSqr;

	int nnz = rowptr[n];

	cudaMalloc((void**)&d_rowPtr, (n + 1) * sizeof(int));
	cudaMalloc((void**)&d_col, nnz * sizeof(int));
	cudaMalloc((void**)&d_val, nnz * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));
	cudaMalloc((void**)&d_Ab, n * sizeof(float));
	cudaMalloc((void**)&d_normAbSqr, sizeof(float));
	cudaMalloc((void**)&d_bAb, sizeof(float));
	cudaMalloc((void**)&d_bb, sizeof(float));
	cudaMalloc((void**)&d_resSqr, sizeof(float));

	cudaMemcpy(d_rowPtr, rowptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, col, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b[0], n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ab, &h_Ab[0], n * sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridSize(2, 1);
	dim3 blockSize(2, 1);
	for (int k = 0; k < maxIter; k++) {
		updateEigenValue << <gridSize, blockSize >> > (d_rowPtr, d_col, d_val, d_b, d_Ab, d_normAbSqr, d_bAb, d_bb, n);

		cudaMemcpy(&h_normAbSqr, d_normAbSqr, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_bAb, d_bAb, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_bb, d_bb, sizeof(float), cudaMemcpyDeviceToHost);

		lambda = h_bAb / h_bb;

		updateEigenVector << <gridSize, blockSize >> > (d_b, d_Ab, d_normAbSqr, n);

		updateResidualSqr << <gridSize, blockSize >> > (d_b, d_Ab, lambda, d_resSqr, n);

		cudaMemcpy(&h_resSqr, d_resSqr, sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "eigen value: " << lambda << " normABSqr: " << h_normAbSqr << " bAb: " << h_bAb << " bb: " << h_bb << " resSqr: " << h_resSqr << std::endl;

		if (h_resSqr < tol) {
			break;
		}

		h_normAbSqr = 0.0f;
		h_bAb = 0.0f;
		h_bb = 0.0f;
		h_resSqr = 0.0f;

		cudaMemcpy(d_normAbSqr, &h_normAbSqr, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bAb, &h_bAb, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bb, &h_bb, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_resSqr, &h_resSqr, sizeof(float), cudaMemcpyHostToDevice);


		/*cudaMemcpy(&h_b[0], d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			std::cout << h_b[i] << " ";
		}
		std::cout << "" << std::endl;*/

		//updateResidualSqr(d_Ab, d_b, lambda, d_resSqr, n);
		// update resSqr
		//resSqr = 0.0;
		//for (int i = 0; i < n; i++) {
		//	resSqr += (temp[i] - lambda * b[i]) * ((temp[i] - lambda * b[i]));
		//}

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

__global__ void updateEigenValue(const int* d_rowPtr, const int* d_col, const float* d_val, const float* d_b, float* d_Ab, float* d_normAbSqr, float* d_bAb, float* d_bb, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	// calculate A*b
	while (index + stride < n) {
		//printf("threadIdx: %d  index: %d  start; %d  end: %d\n", threadIdx.x, index, d_rowPtr[index], d_rowPtr[index + 1]);
		d_Ab[index + stride] = 0.0f;
		for (int i = d_rowPtr[index + stride]; i < d_rowPtr[index + stride + 1]; i++) {
			//printf("threadIdx: %d  i: %d  value: %f\n", threadIdx.x, i, d_val[i]);
			d_Ab[index + stride] += d_val[i] * d_b[d_col[i]];
		}

		stride += blockDim.x * gridDim.x;
	}

	 //calculate dot(temp, temp), dot(b, temp), and dot(b, b)
	__shared__ float cache1[2];  // assumes blockDim.x == 2
	__shared__ float cache2[2];
	__shared__ float cache3[2];

	stride = 0;
	float t1 = 0.0f;
	float t2 = 0.0f;
	float t3 = 0.0f;
	while (index + stride < n) {
		t1 += d_Ab[index + stride] * d_Ab[index + stride];
		t2 += d_b[index + stride] * d_Ab[index + stride];
		t3 += d_b[index + stride] * d_b[index + stride];
		stride += blockDim.x * gridDim.x;
	}

	cache1[threadIdx.x] = t1;
	cache2[threadIdx.x] = t2;
	cache3[threadIdx.x] = t3;

	//printf("threadIdx.x: %d  cache value: %f\n", threadIdx.x, t1);


	// perform reduction on shared cache to get final dot products
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache1[threadIdx.x] += cache1[threadIdx.x + i];
			cache2[threadIdx.x] += cache2[threadIdx.x + i];
			cache3[threadIdx.x] += cache3[threadIdx.x + i];
		}

		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		//printf("cache1[0]: %f  cache2[0]: %f  cache3[0]: %f\n", cache1[0], cache2[0], cache3[0]);
		atomicAdd(d_normAbSqr, cache1[0]);
		atomicAdd(d_bAb, cache2[0]);
		atomicAdd(d_bb, cache3[0]);

		//printf("norm AB: %f  alpha1: %f  alpha2: %f\n", *d_normAb, *d_alpha1, *d_alpha2);
	}
}


__global__ void updateEigenVector(float* d_b, const float* d_Ab, const float* d_normAbSqr, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	float normAb = sqrt(*d_normAbSqr);

	while (index + stride < n) {
		d_b[index] = d_Ab[index] / normAb;

		stride += blockDim.x * gridDim.x;
	}
}

__global__ void updateResidualSqr(const float* d_b, const float* d_Ab, float lambda, float* d_resSqr, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	//resSqr = 0.0;
	//for (int i = 0; i < n; i++) {
	//	resSqr += (temp[i] - lambda * b[i]) * ((temp[i] - lambda * b[i]));
	//}

	__shared__ float cache[2]; // assumes blockDim.x == 2

	float temp = 0.0f;
	while (index + stride < n) {
		float t = d_Ab[index + stride] - lambda * d_b[index + stride];
		temp += t * t;

		stride += blockDim.x * gridDim.x;
	}

	cache[threadIdx.x] = temp;

	// parallel reduction
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] += cache[threadIdx.x + i];

			__syncthreads();
		}

		i /= 2;
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_resSqr, cache[0]);
	}
}