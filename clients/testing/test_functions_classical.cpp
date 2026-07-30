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

bool testing::test_classical(classical_solver solver_type, Arguments arg)
{
    if(arg.backend != backend::CPU)
    {
        std::cout << "Error: Classical solvers only supported on CPU backend." << std::endl;
        return false;
    }

    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);
    mat_A.make_diagonally_dominant();

    // Solution vector
    linalg::vector<double> vec_x(mat_A.get_m());
    vec_x.zeros();

    linalg::vector<double> vec_init_x(mat_A.get_m());
    vec_init_x.zeros();

    // Righthand side vector
    linalg::vector<double> vec_b(mat_A.get_m());
    vec_b.ones();

    linalg::vector<double> vec_e(mat_A.get_n());
    vec_e.ones();

    mat_A.multiply_by_vector(vec_b, vec_e);

    linalg::jacobi_solver jac;
    linalg::gs_solver     gs;
    linalg::sgs_solver    sgs;
    linalg::sor_solver    sor;
    linalg::ssor_solver   ssor;

    switch(solver_type)
    {
    case classical_solver::jacobi:
        jac.build(mat_A);
        break;
    case classical_solver::gauss_seidel:
        gs.build(mat_A);
        break;
    case classical_solver::SOR:
        sor.build(mat_A);
        break;
    case classical_solver::symmetric_gauss_seidel:
        sgs.build(mat_A);
        break;
    case classical_solver::SSOR:
        ssor.build(mat_A);
        break;
    }

    int                  iter = 0;
    linalg::iter_control control;
    control.max_iter = arg.max_iters;

    switch(solver_type)
    {
    case classical_solver::jacobi:
        iter = jac.solve(mat_A, vec_x, vec_b, control);
        break;
    case classical_solver::gauss_seidel:
        iter = gs.solve(mat_A, vec_x, vec_b, control);
        break;
    case classical_solver::SOR:
        iter = sor.solve(mat_A, vec_x, vec_b, control, 0.666667);
        break;
    case classical_solver::symmetric_gauss_seidel:
        iter = sgs.solve(mat_A, vec_x, vec_b, control);
        break;
    case classical_solver::SSOR:
        iter = ssor.solve(mat_A, vec_x, vec_b, control, 0.666667);
        break;
    }

    std::cout << "iter: " << iter << std::endl;

    return check_solution(
        mat_A, vec_b, vec_x, vec_init_x, std::max(control.abs_tol, control.rel_tol), 0);
}
