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
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include "cuda_axpy.h"
#include "cuda_csr2csc.h"
#include "cuda_csrgeam.h"
#include "cuda_csrgemm.h"
#include "cuda_csric0.h"
#include "cuda_csrilu0.h"
#include "cuda_csrtrsv.h"
#include "cuda_extract.h"
#include "cuda_matrix_vector.h"
#include "cuda_scale.h"
#include "cuda_ssor.h"
#include "cuda_tridiagonal.h"

namespace linalg
{
    double cuda_norm_inf(const double* array, int size);
    void   cuda_jacobi_solve(const double* rhs, const double* diag, double* x, size_t size);
}

#endif
