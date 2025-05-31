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

bool Testing::test_amg(AMGSolver solver_type, Arguments arg)
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