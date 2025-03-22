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

#include "test_krylov.h"
#include "utility.h"

#include <cmath>
#include <iostream>
#include <chrono>

#include "linalg.h"

bool Testing::test_krylov(Testing::KrylovSolver solver, Testing::Preconditioner precond, const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    //load_spd_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
    load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);
    std::vector<double> init_x(x);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    std::vector<double> e(n, 1.0);
    matrix_vector_product(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), e.data(), b.data(), n);

    preconditioner* p = nullptr;
    switch(precond)
    {
        case Testing::Preconditioner::Jacobi:
            p = new jacobi_precond;
            break;
        case Testing::Preconditioner::ILU:
            p = new ilu_precond;
            break;
        case Testing::Preconditioner::IC:
            p = new ic_precond;
            break;
    }

    if(p != nullptr)
    {
        p->build(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, n, nnz);
    }

    int iter = 0;

    iter_control control;
    control.max_iter = 5000;

    auto t1 = std::chrono::high_resolution_clock::now();

    switch(solver)
    {
        case Testing::KrylovSolver::CG:
            iter = pcg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, p, control, m);
            //iter = cg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, control, m);
            break;
        case Testing::KrylovSolver::BICGSTAB:
            iter = bicgstab(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, control);
            break;
        case Testing::KrylovSolver::GMRES:
            iter = gmres(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, control, 50);
            break;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << std::endl;

    if(p != nullptr)
    {
        delete p;
    }

    std::cout << "iter: " << iter << std::endl;

    int norm_type = (solver == Testing::KrylovSolver::GMRES) ? 1 : 0;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, init_x, std::max(control.abs_tol, control.rel_tol), norm_type);
}