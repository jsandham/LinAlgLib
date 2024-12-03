#ifndef UTILITY_H__
#define UTILITY_H__

#include <string>
#include <vector>

#include "linalg.h"

bool load_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr, std::vector<int> &csr_col_ind,
                   std::vector<double> &csr_val, int &m, int &n, int &nnz);

bool load_diagonally_dominant_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr,
                                       std::vector<int> &csr_col_ind, std::vector<double> &csr_val, int &m, int &n,
                                       int &nnz);

bool load_spd_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr, std::vector<int> &csr_col_ind,
                       std::vector<double> &csr_val, int &m, int &n, int &nnz);

bool check_solution(const std::vector<int> &csr_row_ptr, const std::vector<int> &csr_col_ind,
                    const std::vector<double> &csr_val, int m, int n, int nnz, const std::vector<double> &b,
                    const std::vector<double> &x, double tol);

#endif