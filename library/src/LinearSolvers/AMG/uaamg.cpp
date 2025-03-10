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

#include "../../../include/LinearSolvers/AMG/uaamg.h"
#include "../../../include/LinearSolvers/AMG/amg_aggregation.h"
#include "../../../include/LinearSolvers/AMG/amg_strength.h"
#include "../../../include/LinearSolvers/slaf.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <vector>

//********************************************************************************
//
// AMG: Unsmoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

static bool construct_prolongation_using_unsmoothed_aggregation(const csr_matrix &A,
                                                                const std::vector<int> &connections,
                                                                const std::vector<int64_t> &aggregates,
                                                                const std::vector<int64_t> &aggregate_root_nodes,
                                                                csr_matrix &prolongation)
{
    prolongation.m = A.m;
    prolongation.nnz = 0;
    prolongation.csr_row_ptr.resize(A.m + 1, 0);

    // Determine number of columns in the prolongation matrix. This will be
    // the maximum aggregate plus one.
    prolongation.n = -1;
    for (size_t i = 0; i < aggregates.size(); i++)
    {
        if (prolongation.n < aggregates[i])
        {
            prolongation.n = aggregates[i];
        }
    }

    prolongation.n++;

    // std::cout << "prolongation.n: " << prolongation.n << " A.m: " << A.m << "
    // A.nnz: " << A.nnz << std::endl;

    std::vector<int> table(prolongation.n, -1);

    // Determine number of non-zeros for P
    for (int i = 0; i < A.m; i++)
    {
        int64_t aggregate = aggregates[i];

        if (aggregate >= 0)
        {
            prolongation.csr_row_ptr[i + 1] = 1;
            prolongation.nnz++;
        }
    }

    // exclusive scan on prolongation row pointer array
    prolongation.csr_row_ptr[0] = 0;
    for (int i = 0; i < prolongation.m; i++)
    {
        prolongation.csr_row_ptr[i + 1] += prolongation.csr_row_ptr[i];
    }

    // std::cout << "prolongation.csr_row_ptr" << std::endl;
    // for (size_t i = 0; i < prolongation.csr_row_ptr.size(); i++)
    //{
    //	std::cout << prolongation.csr_row_ptr[i] << " ";
    // }
    // std::cout << "" << std::endl;

    assert(prolongation.nnz == prolongation.csr_row_ptr[prolongation.m]);

    prolongation.csr_col_ind.resize(prolongation.nnz);
    prolongation.csr_val.resize(prolongation.nnz);

    // Fill P
    for (int i = 0; i < A.m; i++)
    {
        int64_t aggregate = aggregates[i];

        if (aggregate >= 0)
        {
            int start = prolongation.csr_row_ptr[i];

            prolongation.csr_col_ind[start] = aggregate;
            prolongation.csr_val[start] = 1.0;
        }
    }

    return true;
}

static void transpose(const csr_matrix &prolongation, csr_matrix &restriction)
{
    restriction.m = prolongation.n;
    restriction.n = prolongation.m;
    restriction.nnz = prolongation.nnz;
    restriction.csr_row_ptr.resize(restriction.m + 1, 0);
    restriction.csr_col_ind.resize(restriction.nnz, -1);
    restriction.csr_val.resize(restriction.nnz);

    /*std::cout << "prolongation" << std::endl;
    for (int i = 0; i < prolongation.m; i++)
    {
            int row_start = prolongation.csr_row_ptr[i];
            int row_end = prolongation.csr_row_ptr[i + 1];

            std::vector<double> temp(prolongation.n, 0);
            for (int j = row_start; j < row_end; j++)
            {
                    temp[prolongation.csr_col_ind[j]] = prolongation.csr_val[j];
            }

            for (int j = 0; j < prolongation.n; j++)
            {
                    std::cout << temp[j] << " ";
            }
            std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;*/

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

    /*std::cout << "restriction" << std::endl;
    for (int i = 0; i < restriction.m; i++)
    {
            int row_start = restriction.csr_row_ptr[i];
            int row_end = restriction.csr_row_ptr[i + 1];

            std::vector<double> temp(restriction.n, 0);
            for (int j = row_start; j < row_end; j++)
            {
                    temp[restriction.csr_col_ind[j]] = restriction.csr_val[j];
            }

            for (int j = 0; j < restriction.n; j++)
            {
                    std::cout << temp[j] << " ";
            }
            std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;*/
}

static void galarkin_triple_product(const csr_matrix &R, const csr_matrix &A, const csr_matrix &P, csr_matrix &A_coarse)
{
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

    /*std::cout << "A coarse" << std::endl;
    for (int i = 0; i < A_coarse.m; i++)
    {
            int row_start = A_coarse.csr_row_ptr[i];
            int row_end = A_coarse.csr_row_ptr[i + 1];

            std::vector<double> temp(A_coarse.n, 0);
            for (int j = row_start; j < row_end; j++)
            {
                    temp[A_coarse.csr_col_ind[j]] = A_coarse.csr_val[j];
            }

            for (int j = 0; j < A_coarse.n; j++)
            {
                    std::cout << temp[j] << " ";
            }
            std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;*/
}

void uaamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
                 int max_level, heirarchy &hierarchy)
{
    hierarchy.prolongations.resize(max_level);
    hierarchy.restrictions.resize(max_level);
    hierarchy.A_cs.resize(max_level + 1);

    // Set original matrix at level 0 in the hierarchy
    hierarchy.A_cs[0].m = m;
    hierarchy.A_cs[0].n = m;
    hierarchy.A_cs[0].nnz = nnz;
    hierarchy.A_cs[0].csr_row_ptr.resize(m + 1);
    hierarchy.A_cs[0].csr_col_ind.resize(nnz);
    hierarchy.A_cs[0].csr_val.resize(nnz);

    for (int i = 0; i < m + 1; i++)
    {
        hierarchy.A_cs[0].csr_row_ptr[i] = csr_row_ptr[i];
    }

    for (int i = 0; i < nnz; i++)
    {
        hierarchy.A_cs[0].csr_col_ind[i] = csr_col_ind[i];
        hierarchy.A_cs[0].csr_val[i] = csr_val[i];
    }

    double eps = 0.001;

    int level = 0;
    while (level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;

        csr_matrix &A = hierarchy.A_cs[level];
        csr_matrix &A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix &P = hierarchy.prolongations[level];
        csr_matrix &R = hierarchy.restrictions[level];

        std::vector<int> connections;
        std::vector<int64_t> aggregates;
        std::vector<int64_t> aggregate_root_nodes;

        connections.resize(A.nnz, 0);
        aggregates.resize(A.m, 0);

        // Compute strength of connections
        compute_strong_connections(A, eps, connections);

        // Compute aggregations using parallel maximal independent set
        compute_aggregates_using_pmis(A, connections, aggregates, aggregate_root_nodes);

        // Construct prolongation matrix using smoothed aggregation
        construct_prolongation_using_unsmoothed_aggregation(A, connections, aggregates, aggregate_root_nodes, P);

        if (P.n == 0)
        {
            break;
        }

        // Compute restriction matrix by transpose of prolongation matrix
        transpose(P, R);

        // Compute coarse grid matrix using Galarkin triple product A_c = R * A * P
        galarkin_triple_product(R, A, P, A_coarse);

        level++;
        eps *= 0.5;
    }

    hierarchy.total_levels = level;

    std::cout << "Total number of levels in operator hierarchy at the end of the "
                 "setup phase: "
              << level << std::endl;
}
