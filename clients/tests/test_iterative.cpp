#include "test_iterative.h"
#include "utility.h"

#include <cmath>
#include <iostream>

#include "linalg.h"

bool Testing::test_jacobi(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_diagonally_dominant_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    int it = jac(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.00001, 1000);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}

bool Testing::test_gauss_seidel(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_diagonally_dominant_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    int it = gs(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.00001, 1000);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}

bool Testing::test_sor(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_diagonally_dominant_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    int it =
        sor(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.666667, 0.00001, 1000);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}

bool Testing::test_symmetric_gauss_seidel(const std::string &matrix_file)
{
    int m, n, nnz;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<double> csr_val;
    load_spd_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    // std::cout << "A" << std::endl;
    // for (int i = 0; i < m; i++)
    // {
    // 	int row_start = csr_row_ptr[i];
    // 	int row_end = csr_row_ptr[i + 1];

    // 	std::vector<double> temp(n, 0);
    // 	for (int j = row_start; j < row_end; j++)
    // 	{
    // 		temp[csr_col_ind[j]] = csr_val[j];
    // 	}

    // 	for (int j = 0; j < n; j++)
    // 	{
    // 		std::cout << temp[j] << " ";
    // 	}
    // 	std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;

    // Solution vector
    std::vector<double> x(m, 0.0);

    // Righthand side vector
    std::vector<double> b(m, 1.0);

    int it = sgs(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.00001, 1000);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}

bool Testing::test_ssor(const std::string &matrix_file)
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

    int it =
        ssor(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.66667, 0.00001, 1000);

    std::cout << "it: " << it << std::endl;

    return check_solution(csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, b, x, 0.00001);
}