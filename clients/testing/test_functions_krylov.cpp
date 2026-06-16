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

#include <chrono>
#include <cmath>
#include <iostream>

#include "linalg.h"

bool testing::test_krylov(krylov_solver solver_type, Arguments arg)
{
    linalg::csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

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

    linalg::cg_solver       cg;
    linalg::bicgstab_solver bicgstab;
    linalg::gmres_solver    gmres;

    switch(solver_type)
    {
    case krylov_solver::CG:
        cg.build(mat_A);
        break;
    case krylov_solver::BICGSTAB:
        bicgstab.build(mat_A);
        break;
    case krylov_solver::GMRES:
        gmres.build(mat_A, 100);
        break;
    }

    linalg::preconditioner* p = nullptr;
    switch(arg.precond_type)
    {
    case testing::preconditioner::Jacobi:
        p = new linalg::jacobi_precond;
        break;
    case testing::preconditioner::GaussSeidel:
        p = new linalg::gauss_seidel_precond;
        break;
    case testing::preconditioner::SOR:
        p = new linalg::SOR_precond(0.3);
        break;
    case testing::preconditioner::SymmGaussSeidel:
        p = new linalg::symmetric_gauss_seidel_precond;
        break;
    case testing::preconditioner::SSOR:
        p = new linalg::SSOR_precond(1.2);
        break;
    case testing::preconditioner::ILU:
        p = new linalg::ilu_precond;
        break;
    case testing::preconditioner::IC:
        p = new linalg::ic_precond;
        break;
    }

    mat_A.move_to_device();
    vec_x.move_to_device();
    vec_b.move_to_device();
    cg.move_to_device();
    bicgstab.move_to_device();
    if(p != nullptr)
    {
        p->move_to_device();
    }

    if(p != nullptr)
    {
        std::cout << "Build preconditioner" << std::endl;
        p->build(mat_A);
    }

    int iter = 0;

    linalg::iter_control control;
    control.max_iter = arg.max_iters;

    auto t1 = std::chrono::high_resolution_clock::now();

    switch(solver_type)
    {
    case krylov_solver::CG:
        iter = cg.solve(mat_A, vec_x, vec_b, p, control);
        break;
    case krylov_solver::BICGSTAB:
        iter = bicgstab.solve(mat_A, vec_x, vec_b, p, control);
        break;
    case krylov_solver::GMRES:
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

    mat_A.move_to_host();
    vec_x.move_to_host();
    vec_b.move_to_host();
    cg.move_to_host();
    bicgstab.move_to_host();

    std::cout << "iter: " << iter << std::endl;

    int norm_type = (solver_type == krylov_solver::GMRES) ? 1 : 0;

    return check_solution(
        mat_A, vec_b, vec_x, vec_init_x, std::max(control.abs_tol, control.rel_tol), norm_type);
}
