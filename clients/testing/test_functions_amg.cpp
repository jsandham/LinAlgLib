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

bool Testing::test_amg(AMGSolver solver_type, Arguments arg)
{
    csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

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

    int max_levels = 100;

    hierarchy hierachy;
    switch(solver_type)
    {
        case AMGSolver::UAAMG:
            uaamg_setup(mat_A, max_levels, hierachy);
            break;
        case AMGSolver::SAAMG:
            saamg_setup(mat_A, max_levels, hierachy);
            break;
        case AMGSolver::RSAMG:
            rsamg_setup(mat_A, max_levels, hierachy);
            break;
    }

    std::cout << "arg.presmoothing: " << arg.presmoothing 
              << " arg.postsmoothing: " << arg.postsmoothing 
              << " arg.cycle: " << CycleToString(arg.cycle)
              << " arg.smoother: " << SmootherToString(arg.smoother) << std::endl;

    iter_control control;

    // int cycles = amg_solve(hierachy, x.data(), b.data(), arg.presmoothing, arg.postsmoothing, arg.cycle, arg.smoother, control);
    int cycles = amg_solve(hierachy, vec_x, vec_b, arg.presmoothing, arg.postsmoothing, arg.cycle, arg.smoother, control);

    std::cout << "cycles: " << cycles << std::endl;

    int norm_type = 0;

    return check_solution(mat_A, vec_b, vec_x, vec_init_x, std::max(control.abs_tol, control.rel_tol), norm_type);
}