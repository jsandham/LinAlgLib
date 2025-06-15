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

using namespace linalg;

bool Testing::test_classical(ClassicalSolver solver_type, Arguments arg)
{
    csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);
    mat_A.make_diagonally_dominant();

    // Solution vector
    vector<double> vec_x(mat_A.get_m());
    vec_x.zeros();

    vector<double> vec_init_x(mat_A.get_m());
    vec_init_x.zeros();

    // Righthand side vector
    vector<double> vec_b(mat_A.get_m());
    vec_b.ones();

    vector<double> vec_e(mat_A.get_n());
    vec_e.ones();

    mat_A.multiply_by_vector(vec_b, vec_e);

    std::cout << "mat_A.get_m(): " << mat_A.get_m()
              << " mat_A.get_n(): " << mat_A.get_n()
              << " mat_A.get_nnz(): " << mat_A.get_nnz() << std::endl; 

    jacobi_solver jac;
    gs_solver gs;
    sgs_solver sgs;
    sor_solver sor;
    ssor_solver ssor;   

    switch(solver_type)
    {
        case ClassicalSolver::Jacobi:
            jac.build(mat_A);
            break;
        case ClassicalSolver::GaussSeidel:
            gs.build(mat_A);
            break;
        case ClassicalSolver::SOR:
            sor.build(mat_A);
            break;
        case ClassicalSolver::SymmGaussSeidel:
            sgs.build(mat_A);
            break;
        case ClassicalSolver::SSOR:
            ssor.build(mat_A);
            break;
    }

    int iter = 0;
    iter_control control;
    control.max_iter = arg.max_iters;

    switch(solver_type)
    {
        case ClassicalSolver::Jacobi:
            iter = jac.solve(mat_A, vec_x, vec_b, control);
            break;
        case ClassicalSolver::GaussSeidel:
            iter = gs.solve(mat_A, vec_x, vec_b, control);
            break;
        case ClassicalSolver::SOR:
            iter = sor.solve(mat_A, vec_x, vec_b, control, 0.666667);
            break;
        case ClassicalSolver::SymmGaussSeidel:
            iter = sgs.solve(mat_A, vec_x, vec_b, control);
            break;
        case ClassicalSolver::SSOR:
            iter = ssor.solve(mat_A, vec_x, vec_b, control, 0.666667);
            break;
    }

    std::cout << "iter: " << iter << std::endl;

    return check_solution(mat_A, vec_b, vec_x, vec_init_x, std::max(control.abs_tol, control.rel_tol), 0);
}