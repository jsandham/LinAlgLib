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
#include <chrono>

#include "linalg.h"

bool Testing::test_krylov(KrylovSolver solver_type, Arguments arg)
{
    csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    // Solution vector
    vector vec_x(mat_A.get_m());
    vec_x.zeros();

    vector vec_init_x(mat_A.get_m());
    vec_init_x.zeros();

    // Righthand side vector
    vector vec_b(mat_A.get_m());
    vec_b.ones();

    vector vec_e(mat_A.get_n());
    vec_e.ones();

    mat_A.multiply_vector(vec_b, vec_e);

    cg_solver cg;
    bicgstab_solver bicgstab;
    gmres_solver gmres;

    switch(solver_type)
    {
        case KrylovSolver::CG:
            cg.build(mat_A);
            break;
        case KrylovSolver::BICGSTAB:
            bicgstab.build(mat_A);
            break;
        case KrylovSolver::GMRES:
            gmres.build(mat_A, 100);
            break;
    }

    preconditioner* p = nullptr;
    switch(arg.precond)
    {
        case Testing::Preconditioner::Jacobi:
            p = new jacobi_precond;
            break;
        case Testing::Preconditioner::GaussSeidel:
            p = new gauss_seidel_precond;
            break;
        case Testing::Preconditioner::SOR:
            p = new SOR_precond(0.3);
            break;
        case Testing::Preconditioner::SymmGaussSeidel:
            p = new symmetric_gauss_seidel_precond;
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
        std::cout << "Build preconditioner" << std::endl;
        p->build(mat_A);
    }

    int iter = 0;

    iter_control control;
    control.max_iter = arg.max_iters;

    auto t1 = std::chrono::high_resolution_clock::now();

    switch(solver_type)
    {
        case KrylovSolver::CG:
            iter = cg.solve(mat_A, vec_x, vec_b, p, control);
            break;
        case KrylovSolver::BICGSTAB:
            iter = bicgstab.solve(mat_A, vec_x, vec_b, p, control);
            break;
        case KrylovSolver::GMRES:
            iter = gmres.solve(mat_A, vec_x, vec_b, p, control);
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

    int norm_type = (solver_type == KrylovSolver::GMRES) ? 1 : 0;

    int m = mat_A.get_m();
    int n = mat_A.get_n();
    int nnz = mat_A.get_nnz();
    std::vector<int> csr_row_ptr(mat_A.get_row_ptr(), mat_A.get_row_ptr() + (m + 1));
    std::vector<int> csr_col_ind(mat_A.get_col_ind(), mat_A.get_col_ind() + nnz);
    std::vector<double> csr_val(mat_A.get_val(), mat_A.get_val() + nnz);
    std::vector<double> b(vec_b.get_vec(), vec_b.get_vec() + m);
    std::vector<double> x(vec_x.get_vec(), vec_x.get_vec() + n);
    std::vector<double> init_x(vec_init_x.get_vec(), vec_init_x.get_vec() + n);

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, init_x, std::max(control.abs_tol, control.rel_tol), norm_type);
}