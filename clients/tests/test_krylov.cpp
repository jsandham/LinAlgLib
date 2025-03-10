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

#include "linalg.h"

bool Testing::test_krylov(Testing::KrylovSolver solver, Testing::Preconditioner precond, const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    //load_spd_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
    load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    for(int i = 0; i < nnz; i++)
    {
        csr_val[i] = 1.0;
    }


    std::cout << "A" << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        std::vector<double> temp(n, 0.0);
        for(int j = start; j < end; j++)
        {
            temp[csr_col_ind[j]] = csr_val[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;




    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

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
    int max_iter = 1;//5000;
    int restart_iter = 4;//1000;
    double tol = 1e-8;

    switch(solver)
    {
        case Testing::KrylovSolver::CG:
            iter = cg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter, restart_iter);
            break;
        case Testing::KrylovSolver::BICGSTAB:
            iter = bicgstab(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter);
            break;
        case Testing::KrylovSolver::GMRES:
            iter = gmres(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, tol, max_iter, restart_iter);
            break;
    }

    if(p != nullptr)
    {
        delete p;
    }

    std::cout << "iter: " << iter << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, tol);
}