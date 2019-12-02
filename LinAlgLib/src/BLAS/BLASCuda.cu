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

#include "../../include/BLAS/BLASCuda.cuh"
#include "../../include/BLAS/BLASCudaKernels.cuh"

void srot(float* d_x, float* d_y, float s, float c, int n)
{

}

void drot(double* d_x, double* d_y, double s, double c, int n)
{

}

void sswap(float* d_x, float* d_y, int n)
{

}

void dswap(double* d_x, double* d_y, int n)
{

}

void sscal(float* d_x, float alpha, int n)
{

}

void dscal(double* d_x, double alpha, int n)
{

}

void scopy(const float* d_c, float* d_y, int n)
{

}

void dcopy(const double* d_c, double* d_y, int n)
{

}

void saxpy(const float* d_x, float* d_y, float alpha, int n)
{

}

void daxpy(const double* d_x, double* d_y, double alpha, int n)
{

}

void sdot(const float* d_x, const float* d_y, float* d_product, int n)
{

}

void ddot(const double* d_x, const double* d_y, double* d_product, int n)
{

}

void sasum(const float* d_x, float* d_sum, int n)
{

}

void dasum(const double* d_x, double* d_sum, int n)
{

}

void samin(const float* d_x, int* d_aminIndex, int n)
{

}

void damin(const double* d_x, int* d_aminIndex, int n)
{

}

void samax(const float* d_x, int* d_amaxIndex, int n)
{

}

void damax(const double* d_x, int* d_amaxIndex, int n)
{

}

void sgemv(const float* d_A, const float* d_x, float* d_y, float alpha, float beta, int nrow, int ncol)
{

}

void dgemv(const double* d_A, const double* d_x, double* d_y, double alpha, double beta, int nrow, int ncol)
{

}

void sgemm(const float* d_A, const float* d_B, float* d_C, float alpha, float beta, int nrow, int ncol)
{

}

void dgemm(const double* d_A, const double* d_B, double* d_C, double alpha, double beta, int nrow, int ncol)
{

}