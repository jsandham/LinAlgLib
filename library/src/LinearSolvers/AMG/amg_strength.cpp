//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include "../../../include/LinearSolvers/AMG/amg_strength.h"
#include <assert.h>
#include <cmath>
#include <iostream>

static void extract_diagonal(const csr_matrix &A, std::vector<double> &diag)
{
    assert(A.m == diag.size());

    for (int i = 0; i < A.m; i++)
    {
        int row_start = A.csr_row_ptr[i];
        int row_end = A.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (A.csr_col_ind[j] == i)
            {
                diag[i] = A.csr_val[j];
                break;
            }
        }
    }
}

void compute_strong_connections(const csr_matrix &A, double eps, std::vector<int> &connections)
{
    // Extract diagaonl
    std::vector<double> diag(A.m);
    extract_diagonal(A, diag);

    // std::cout << "diag" << std::endl;
    // for (size_t i = 0; i < diag.size(); i++)
    // {
    // 	std::cout << diag[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // double eps2 = eps * eps;

    for (int i = 0; i < A.m; i++)
    {
        // double eps_dia_i = eps2 * diag[i];

        int row_start = A.csr_row_ptr[i];
        int row_end = A.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            int c = A.csr_col_ind[j];
            double v = A.csr_val[j];

            assert(c >= 0);
            assert(c < A.m);

            // connections[j] = (c != i) && (v * v > eps_dia_i * diag[c]);
            connections[j] = (c != i) && (std::abs(v) >= eps * std::sqrt(std::abs(diag[i]) * std::abs(diag[c])));
        }
    }
}