#include "test_rsamg.h"
#include "utility.h"

#include <cmath>
#include <iostream>

#include "linalg.h"

bool Testing::test_rsamg(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    heirarchy hierachy;
    rsamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 10, hierachy);

    int cycles = amg_solve(hierachy, x.data(), b.data(), 10, 10, 0.00001, Cycle::Vcycle, Smoother::Gauss_Siedel);
    // int cycles = amg_solve(hierachy, x.data(), b.data(), 2, 2, 0.00001, Cycle::Wcycle, Smoother::Gauss_Siedel);
    // int cycles = amg_solve(hierachy, x.data(), b.data(), 2, 2, 0.00001, Cycle::Wcycle, Smoother::SOR);

    std::cout << "cycles: " << cycles << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}