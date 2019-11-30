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

float powerIterationCuda(const int rowptr[], const int col[], const float val[], float eigenVec[], const int n, const int maxIter)
{
	//std::vector<double> b;
	//std::vector<double> temp;
	//b.resize(n);
	//temp.resize(n);

	//std::default_random_engine generator;
	//std::uniform_real_distribution<double> distribution(0.0, 1.0);

	//for (int i = 0; i < n; i++) {
	//	double number = distribution(generator);
	//	b[i] = 2.0 * number - 1.0;
	//	temp[i] = 0.0;
	//}

	//double eigenValue = 0.0;

	//for (int k = 0; k < maxIter; k++) {

	//	
	//}

	//// fill in eigen vector
	//for (int i = 0; i < n; i++) {
	//	eigenVec[i] = b[i];
	//	//std::cout << eigenVec[i] << std::endl;
	//}

	//return eigenValue;

	return 0.0f;
}

__global__ void updateEigenValue(int* d_rowPtr, int* d_col, float* d_val, float* d_b, float* d_temp, float* d_normAb, float* d_alpha1, float* d_alpha2, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	// calculate A*b
	while (index + stride < n) {
		printf("threadIdx: %d  index: %d  start; %d  end: %d\n", threadIdx.x, index, d_rowPtr[index], d_rowPtr[index + 1]);
		for (int i = d_rowPtr[index]; i < d_rowPtr[index + 1]; i++) {
			printf("threadIdx: %d  i: %d  value: %f\n", threadIdx.x, i, d_val[i]);
			d_temp[index] += d_val[i] * d_b[d_col[i]];
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
		t1 += d_temp[index + stride] * d_temp[index + stride];
		t2 += d_b[index + stride] * d_temp[index + stride];
		t3 += d_b[index + stride] * d_b[index + stride];
		stride += blockDim.x * gridDim.x;
	}

	cache1[threadIdx.x] = t1;
	cache2[threadIdx.x] = t2;
	cache3[threadIdx.x] = t3;

	printf("threadIdx.x: %d  cache value: %f\n", threadIdx.x, t1);


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
		printf("cache1[0]: %f  cache2[0]: %f  cache3[0]: %f\n", cache1[0], cache2[0], cache3[0]);
		atomicAdd(d_normAb, cache1[0]);
		atomicAdd(d_alpha1, cache2[0]);
		atomicAdd(d_alpha2, cache3[0]);

		printf("norm AB: %f  alpha1: %f  alpha2: %f\n", *d_normAb, *d_alpha1, *d_alpha2);
	}

	//// update b
	//for (int i = 0; i < n; i++) {
	//	b[i] = temp[i] / normAb;
	//}
}


__global__ void updateEigenVector(float* d_b, float* d_temp, float* normAb, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 0;

	while (index + stride < n) {
		d_b[index] = d_temp[index] / *normAb;

		stride += blockDim.x * gridDim.x;
	}
}