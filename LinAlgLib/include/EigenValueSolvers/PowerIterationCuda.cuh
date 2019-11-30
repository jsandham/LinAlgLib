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

float powerIterationCuda(const int* rowptr, const int* col, const float* val, float* eigenVec, float tol, int n, int maxIter);

__global__ void updateEigenValue(const int* d_rowPtr, const int* d_col, const float* d_val, const float* d_b, float* d_Ab, float* d_normAbSqr, float* d_bAb, float* d_bb, int n);
__global__ void updateEigenVector(float* d_b, const float* d_Ab, const float* d_normAbSqr, int n);
__global__ void updateResidualSqr(const float* d_b, const float* d_Ab, float lambda, float* d_resSqr, int n);

#endif