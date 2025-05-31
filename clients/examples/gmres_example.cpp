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

#include <iostream>
#include <vector>

#include "linalg.h"

int main()
{
    int m = 5;
    int n = 5;
    int nnz = 15;

    // 4 3 0 0 2
    // 1 2 3 0 0
    // 0 5 4 3 0
    // 0 0 1 3 2
    // 9 0 0 6 7
    std::vector<int> csr_row_ptr = {0, 3, 6, 9, 12, 15};
    std::vector<int> csr_col_ind = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
    std::vector<double> csr_val = {4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 1.0, 3.0, 2.0, 9.0, 6.0, 7.0};

    csr_matrix A(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    vector x(A.get_m());
    x.zeros();

    // Righthand side vector
    vector b(A.get_m());
    b.ones();

    gmres_solver gmres;
    gmres.build(A, 5);

    iter_control control;

    int iter = gmres.solve(A, x, b, nullptr, control);

    std::cout << "iter: " << iter << std::endl;

    // Print solution
    std::cout << "x" << std::endl;
    for (int i = 0; i < x.get_size(); i++)
    {
        std::cout << x[i] << " ";
    }
    std::cout << "" << std::endl;

    return 0;
}