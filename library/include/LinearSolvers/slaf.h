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

#ifndef SLAF_H
#define SLAF_H

void matrixVectorProduct(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                         double *y, const int n);

double dotProduct(const double *x, const double *y, const int n);

void diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, const int n);

void forwardSolve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                  const int n);

void backwardSolve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   const int n);

double error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             const int n);

double fast_error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                  const double *b, const int n, const double tol);

#endif
