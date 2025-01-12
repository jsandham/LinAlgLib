#include "test_pcg.h"
#include "utility.h"

#include <cmath>
#include <iostream>

#include "linalg.h"

bool Testing::test_pcg(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_spd_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    // // Jacobi preconditioner
    //jacobi_precond precond;
    //ilu_precond precond;
    //ic_precond precond;

    //precond.build(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, n, nnz);

    //int it = pcg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, &precond, 0.00001, 1000, 1000);
    int it = cg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.00001, 1000, 10);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}