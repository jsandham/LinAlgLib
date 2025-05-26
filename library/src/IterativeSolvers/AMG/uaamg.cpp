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

#include "../../../include/IterativeSolvers/AMG/uaamg.h"
#include "../../../include/IterativeSolvers/AMG/amg_aggregation.h"
#include "../../../include/IterativeSolvers/AMG/amg_strength.h"
#include "../../../include/IterativeSolvers/AMG/amg_util.h"
#include "../../../include/slaf.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>

#include "../../trace.h"

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
    ROUTINE_TRACE("construct_prolongation_using_unsmoothed_aggregation");

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

void uaamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
                 int max_level, heirarchy &hierarchy)
{
    ROUTINE_TRACE("uaamg_setup");

    auto t1 = std::chrono::high_resolution_clock::now();

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

        const csr_matrix &A = hierarchy.A_cs[level];
        csr_matrix &A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix &P = hierarchy.prolongations[level];
        csr_matrix &R = hierarchy.restrictions[level];

        print_matrix("A", A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), A.m, A.n, A.nnz);

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

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Unsmoothed Aggregation AMG setup time: " << ms_double.count() << "ms" << std::endl;
}

void uaamg_setup(const csr_matrix2& A, int max_level, heirarchy &hierarchy)
{
    uaamg_setup(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), A.get_n(), A.get_nnz(),
                 max_level, hierarchy);
}