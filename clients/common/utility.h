#ifndef UTILITY_H__
#define UTILITY_H__

#include <string>
#include <vector>

bool load_mtx_file(const std::string& filename, std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_ind, std::vector<double>& csr_val);

#endif