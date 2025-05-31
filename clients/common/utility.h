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

// bool load_spd_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr, std::vector<int> &csr_col_ind,
//                        std::vector<double> &csr_val, int &m, int &n, int &nnz);

bool check_solution(const std::vector<int> &csr_row_ptr, const std::vector<int> &csr_col_ind,
                    const std::vector<double> &csr_val, int m, int n, int nnz, const std::vector<double> &b,
                    const std::vector<double> &x, const std::vector<double> &init_x, double tol, int norm);

#endif