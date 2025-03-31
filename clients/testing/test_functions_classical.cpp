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

#include "test_functions.h"
#include "utility.h"

#include <cmath>
#include <iostream>

#include "linalg.h"

bool Testing::test_classical(ClassicalSolver solver, Arguments arg)
{
    std::cout << "Testing::test_classical(Arguments arg)" << std::endl;
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_diagonally_dominant_mtx_file(arg.filename, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);
    std::vector<double> init_x = x;

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    int iter = 0;
    int max_iter = 1000;
    double tol = 1e-8;
    
    switch(solver)
    {
        case ClassicalSolver::Jacobi:
            iter = jacobi(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter);
            break;
        case ClassicalSolver::GaussSeidel:
            iter = gs(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter);
            break;
        case ClassicalSolver::SOR:
            iter = sor(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.666667, tol, max_iter);
            break;
        case ClassicalSolver::SymmGaussSeidel:
            iter = sgs(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter);
            break;
        case ClassicalSolver::SSOR:
            iter = ssor(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.666667, tol, max_iter);
            break;
    }

    std::cout << "iter: " << iter << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, init_x, tol, 0);
}