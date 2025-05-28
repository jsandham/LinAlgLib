//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

#include "../../../include/IterativeSolvers/AMG/amg_util.h"
#include "../../../include/slaf.h"

#include "../../trace.h"

void transpose(const csr_matrix &prolongation, csr_matrix &restriction)
{
    ROUTINE_TRACE("transpose");

    restriction.m = prolongation.n;
    restriction.n = prolongation.m;
    restriction.nnz = prolongation.nnz;
    restriction.csr_row_ptr.resize(restriction.m + 1);
    restriction.csr_col_ind.resize(restriction.nnz);
    restriction.csr_val.resize(restriction.nnz);

    // Fill arrays
    for (size_t i = 0; i < restriction.csr_row_ptr.size(); i++)
    {
        restriction.csr_row_ptr[i] = 0;
    }

    for (size_t i = 0; i < restriction.csr_col_ind.size(); i++)
    {
        restriction.csr_col_ind[i] = -1;
    }

    // std::cout << "prolongation" << std::endl;
    // for (int i = 0; i < prolongation.m; i++)
    // {
    // 	int row_start = prolongation.csr_row_ptr[i];
    // 	int row_end = prolongation.csr_row_ptr[i + 1];

    // 	std::vector<double> temp(prolongation.n, 0);
    // 	for (int j = row_start; j < row_end; j++)
    // 	{
    // 		temp[prolongation.csr_col_ind[j]] = prolongation.csr_val[j];
    // 	}

    // 	for (int j = 0; j < prolongation.n; j++)
    // 	{
    // 		std::cout << temp[j] << " ";
    // 	}
    // 	std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;

    for (int i = 0; i < prolongation.m; i++)
    {
        int row_start = prolongation.csr_row_ptr[i];
        int row_end = prolongation.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            restriction.csr_row_ptr[prolongation.csr_col_ind[j] + 1]++;
        }
    }

    // Exclusive scan on row pointer array
    for (int i = 0; i < restriction.m; i++)
    {
        restriction.csr_row_ptr[i + 1] += restriction.csr_row_ptr[i];
    }

    for (int i = 0; i < prolongation.m; i++)
    {
        int row_start = prolongation.csr_row_ptr[i];
        int row_end = prolongation.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            int col = prolongation.csr_col_ind[j];
            double val = prolongation.csr_val[j];

            int start = restriction.csr_row_ptr[col];
            int end = restriction.csr_row_ptr[col + 1];

            for (int k = start; k < end; k++)
            {
                if (restriction.csr_col_ind[k] == -1)
                {
                    restriction.csr_col_ind[k] = i;
                    restriction.csr_val[k] = val;
                    break;
                }
            }
        }
    }

    // std::cout << "restriction" << std::endl;
    // for (int i = 0; i < restriction.m; i++)
    // {
    // 	int row_start = restriction.csr_row_ptr[i];
    // 	int row_end = restriction.csr_row_ptr[i + 1];

    // 	std::vector<double> temp(restriction.n, 0);
    // 	for (int j = row_start; j < row_end; j++)
    // 	{
    // 		temp[restriction.csr_col_ind[j]] = restriction.csr_val[j];
    // 	}

    // 	for (int j = 0; j < restriction.n; j++)
    // 	{
    // 		std::cout << temp[j] << " ";
    // 	}
    // 	std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;
}

void galarkin_triple_product(const csr_matrix &R, const csr_matrix &A, const csr_matrix &P, csr_matrix &A_coarse)
{
    ROUTINE_TRACE("galarkin_triple_product");

    // Compute A_c = R * A * P
    double alpha = 1.0;
    double beta = 0.0;

    // Determine number of non-zeros in A * P product
    csr_matrix AP;
    AP.m = A.m;
    AP.n = P.n;
    AP.nnz = 0;
    AP.csr_row_ptr.resize(AP.m + 1, 0);

    csrgemm_nnz(A.m, P.n, A.n, A.nnz, P.nnz, 0, alpha, A.csr_row_ptr.data(), A.csr_col_ind.data(), P.csr_row_ptr.data(),
                P.csr_col_ind.data(), beta, nullptr, nullptr, AP.csr_row_ptr.data(), &AP.nnz);

    // std::cout << "AP.nnz: " << AP.nnz << std::endl;
    AP.csr_col_ind.resize(AP.nnz);
    AP.csr_val.resize(AP.nnz);

    csrgemm(A.m, P.n, A.n, A.nnz, P.nnz, 0, alpha, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(),
            P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), beta, nullptr, nullptr, nullptr,
            AP.csr_row_ptr.data(), AP.csr_col_ind.data(), AP.csr_val.data());

    // Determine number of non-zeros in A_coarse = R * AP product
    A_coarse.m = R.m;
    A_coarse.n = AP.n;
    A_coarse.nnz = 0;
    A_coarse.csr_row_ptr.resize(A_coarse.m + 1, 0);

    csrgemm_nnz(R.m, AP.n, R.n, R.nnz, AP.nnz, 0, alpha, R.csr_row_ptr.data(), R.csr_col_ind.data(),
                AP.csr_row_ptr.data(), AP.csr_col_ind.data(), beta, nullptr, nullptr, A_coarse.csr_row_ptr.data(),
                &A_coarse.nnz);

    // std::cout << "A_coarse.nnz: " << A_coarse.nnz << std::endl;
    A_coarse.csr_col_ind.resize(A_coarse.nnz);
    A_coarse.csr_val.resize(A_coarse.nnz);

    csrgemm(R.m, AP.n, R.n, R.nnz, AP.nnz, 0, alpha, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(),
            AP.csr_row_ptr.data(), AP.csr_col_ind.data(), AP.csr_val.data(), beta, nullptr, nullptr, nullptr,
            A_coarse.csr_row_ptr.data(), A_coarse.csr_col_ind.data(), A_coarse.csr_val.data());

    // std::cout << "A coarse" << std::endl;
    // for (int i = 0; i < A_coarse.m; i++)
    // {
    // 	int row_start = A_coarse.csr_row_ptr[i];
    // 	int row_end = A_coarse.csr_row_ptr[i + 1];

    // 	std::vector<double> temp(A_coarse.n, 0);
    // 	for (int j = row_start; j < row_end; j++)
    // 	{
    // 		temp[A_coarse.csr_col_ind[j]] = A_coarse.csr_val[j];
    // 	}

    // 	for (int j = 0; j < A_coarse.n; j++)
    // 	{
    // 		std::cout << temp[j] << " ";
    // 	}
    // 	std::cout << "" << std::endl;
    // }
    // std::cout << "" << std::endl;
}

void galarkin_triple_product(const csr_matrix2 &R, const csr_matrix2 &A, const csr_matrix2 &P, csr_matrix2 &A_coarse)
{
    ROUTINE_TRACE("galarkin_triple_product");

    // Compute AP = A * P;
    csr_matrix2 AP;
    A.multiply_matrix(AP, P);

    // Compute A_coarse = R * A * P
    R.multiply_matrix(A_coarse, AP);

    //A_coarse.print_matrix("A coarse");
}

