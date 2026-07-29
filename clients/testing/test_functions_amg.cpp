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

bool testing::test_amg(AMG_solver solver_type, Arguments arg)
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

    linalg::hierarchy hierachy;

    // mat_A.move_to_device();
    // vec_x.move_to_device();
    // vec_init_x.move_to_device();
    // vec_b.move_to_device();
    // vec_e.move_to_device();
    // hierachy.move_to_device();

    mat_A.multiply_by_vector(vec_b, vec_e);

    int max_levels = 100;

    switch(solver_type)
    {
    case AMG_solver::UAAMG:
        uaamg_setup(mat_A, max_levels, hierachy);
        break;
    case AMG_solver::SAAMG:
        saamg_setup(mat_A, max_levels, hierachy);
        break;
    case AMG_solver::RSAMG:
        rsamg_setup(mat_A, max_levels, hierachy);
        break;
    }

    linalg::cycle cycle = linalg::cycle::vcycle;
    switch(arg.cycle_type)
    {
    case testing::cycle_type::vcycle:
        cycle = linalg::cycle::vcycle;
        break;
    case testing::cycle_type::wcycle:
        cycle = linalg::cycle::wcycle;
        break;
    case testing::cycle_type::fcycle:
        cycle = linalg::cycle::fcycle;
        break;
    }

    linalg::smoother smoother = linalg::smoother::jacobi;
    switch(arg.smoother_type)
    {
    case testing::smoother_type::jacobi:
        smoother = linalg::smoother::jacobi;
        break;
    case testing::smoother_type::gauss_seidel:
        smoother = linalg::smoother::gauss_seidel;
        break;
    case testing::smoother_type::symmetric_gauss_seidel:
        smoother = linalg::smoother::symmetric_gauss_seidel;
        break;
    case testing::smoother_type::SOR:
        smoother = linalg::smoother::SOR;
        break;
    case testing::smoother_type::SSOR:
        smoother = linalg::smoother::SSOR;
        break;
    }

    // mat_A.move_to_host();
    // vec_x.move_to_host();
    // vec_init_x.move_to_host();
    // vec_b.move_to_host();
    // vec_e.move_to_host();
    // hierachy.move_to_host();

    std::cout << "arg.presmoothing: " << arg.presmoothing
              << " arg.postsmoothing: " << arg.postsmoothing
              << " arg.cycle: " << cycle_type_to_string(arg.cycle_type)
              << " arg.smoother: " << smoother_type_to_string(arg.smoother_type) << std::endl;

    linalg::iter_control control;
    control.max_cycle = arg.max_iters;

    // int cycles = amg_solve(hierachy, x.data(), b.data(), arg.presmoothing, arg.postsmoothing, arg.cycle, arg.smoother, control);
    int cycles = amg_solve(
        hierachy, vec_x, vec_b, arg.presmoothing, arg.postsmoothing, cycle, smoother, control);

    std::cout << "cycles: " << cycles << std::endl;

    int norm_type = 0;

    return check_solution(
        mat_A, vec_b, vec_x, vec_init_x, std::max(control.abs_tol, control.rel_tol), norm_type);
}
