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

#ifndef HOST_MATH_H
#define HOST_MATH_H

#include <string>

#include "linalg_export.h"
#include "scalar.h"
#include "vector.h"
#include "csr_matrix.h"

namespace linalg
{
    namespace host
    {
        // Compute y = alpha * x + y
        void axpy(double alpha, const vector<double>& x, vector<double>& y);

        // Compute y = alpha * x + y
        void axpy(const scalar<double>& alpha, const vector<double>& x, vector<double>& y);

        // Compute y = alpha * x + beta * y
        void axpby(double alpha, const vector<double>& x, double beta, vector<double>& y);

        // Compute y = alpha * x + beta * y
        void axpby(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, vector<double>& y);

        // Compute z = alpha * x + beta * y + gamma * z
        void axpbypgz(double alpha, const vector<double>& x, double beta, const vector<double>& y, double gamma, vector<double>& z);

        // Compute z = alpha * x + beta * y + gamma * z
        void axpbypgz(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, const vector<double>& y, const scalar<double>& gamma, vector<double>& z);

        // Compute y = A * x
        void matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>&y);

        // Compute y = alpha * A * x + beta * y
        void matrix_vector_product(double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>&y);

        // Compute C = A * B
        void matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

        // Compute C = A + B
        void matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

        // Incomplete IC factorization
        void csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero);

        // Incomplete LU factorization
        void csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero);

        // Forward solve
        void forward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag);

        // Backward solve
        void backward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag);

        // Transpose matrix
        void transpose_matrix(const csr_matrix &A, csr_matrix &transposeA);

        // Dot product
        double dot_product(const vector<double>& x, const vector<double>& y);

        // Dot product
        void dot_product(const vector<double>& x, const vector<double>& y, scalar<double>& result);

        // Compute residual
        void compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res);

        // Exclusive scan
        void exclusize_scan(vector<double>& x);

        // Extract diagonal entries
        void diagonal(const csr_matrix& A, vector<double>& d);

        // Euclidean norm
        double norm_euclid(const vector<double>& array);

        // Infinity norm
        double norm_inf(const vector<double>& array);

        // Fill array with zeros
        void fill_with_zeros(vector<uint32_t> &vec);
        void fill_with_zeros(vector<int32_t> &vec);
        void fill_with_zeros(vector<int64_t> &vec);
        void fill_with_zeros(vector<double> &vec);

        // Fill array with ones
        void fill_with_ones(vector<uint32_t> &vec);
        void fill_with_ones(vector<int32_t> &vec);
        void fill_with_ones(vector<int64_t> &vec);
        void fill_with_ones(vector<double> &vec);

        // Copy array
        void copy(vector<uint32_t> &dest, const vector<uint32_t> &src);
        void copy(vector<int32_t> &dest, const vector<int32_t> &src);
        void copy(vector<int64_t> &dest, const vector<int64_t> &src);
        void copy(vector<double> &dest, const vector<double> &src);
    }
}

#endif
