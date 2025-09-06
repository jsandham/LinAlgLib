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

#include "../../../include/iterative_solvers/amg/uaamg.h"
#include "../../../include/iterative_solvers/amg/amg_aggregation.h"
#include "../../../include/iterative_solvers/amg/amg_strength.h"
#include "../../../include/iterative_solvers/amg/amg_util.h"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../trace.h"

//********************************************************************************
//
// amg: Unsmoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

namespace linalg
{
    static bool construct_prolongation_using_unsmoothed_aggregation(
        const csr_matrix&      A,
        const vector<int>&     connections,
        const vector<int64_t>& aggregates,
        const vector<int64_t>& aggregate_root_nodes,
        csr_matrix&            prolongation)
    {
        ROUTINE_TRACE("construct_prolongation_using_unsmoothed_aggregation");

        // Determine number of columns in the prolongation matrix. This will be
        // the maximum aggregate plus one.
        int64_t n = -1;
        for(size_t i = 0; i < aggregates.get_size(); i++)
        {
            if(n < aggregates[i])
            {
                n = aggregates[i];
            }
        }
        n++;

        prolongation.resize(A.get_m(), (int)n, 0);

        std::cout << "prolongation.n: " << prolongation.get_n() << " A.m: " << A.get_m()
                  << "A.nnz: " << A.get_nnz() << std::endl;

        int* csr_row_ptr_P = prolongation.get_row_ptr();

        std::vector<int> table(prolongation.get_n(), -1);

        // Determine number of non-zeros for P
        int nnz = 0;
        for(int i = 0; i < A.get_m(); i++)
        {
            int64_t aggregate = aggregates[i];

            if(aggregate >= 0)
            {
                csr_row_ptr_P[i + 1] = 1;
                nnz++;
            }
        }

        // exclusive scan on prolongation row pointer array
        csr_row_ptr_P[0] = 0;
        for(int i = 0; i < prolongation.get_m(); i++)
        {
            csr_row_ptr_P[i + 1] += csr_row_ptr_P[i];
        }

        // std::cout << "prolongation.csr_row_ptr" << std::endl;
        // for (int i = 0; i < prolongation.get_m() + 1; i++)
        // {
        // 	std::cout << csr_row_ptr_P[i] << " ";
        // }
        // std::cout << "" << std::endl;

        assert(nnz == csr_row_ptr_P[prolongation.get_m()]);

        prolongation.resize(prolongation.get_m(), prolongation.get_n(), nnz);

        int*    csr_col_ind_P = prolongation.get_col_ind();
        double* csr_val_P     = prolongation.get_val();

        // Fill P
        for(int i = 0; i < A.get_m(); i++)
        {
            int64_t aggregate = aggregates[i];

            if(aggregate >= 0)
            {
                int start = csr_row_ptr_P[i];

                csr_col_ind_P[start] = (int)aggregate;
                csr_val_P[start]     = 1.0;
            }
        }

        return true;
    }
}

void linalg::uaamg_setup(const csr_matrix& A, int max_level, hierarchy& hierarchy)
{
    ROUTINE_TRACE("uaamg_setup");

    auto t1 = std::chrono::high_resolution_clock::now();

    max_level = std::min(max_level, 20);
    //hierarchy.prolongations.resize(max_level);
    //hierarchy.restrictions.resize(max_level);
    //hierarchy.A_cs.resize(max_level + 1);

    // Set original matrix at level 0 in the hierarchy
    hierarchy.A_cs[0].copy_from(A);

    double eps = 0.001;

    vector<int>     connections;
    vector<int64_t> aggregates;
    vector<int64_t> aggregate_root_nodes;

    if(!A.is_on_host())
    {
        connections.move_to_device();
        aggregates.move_to_device();
        aggregate_root_nodes.move_to_device();
    }

    int level = 0;
    while(level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;

        const csr_matrix& A_fine   = hierarchy.A_cs[level];
        csr_matrix&       A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix&       P        = hierarchy.prolongations[level];
        csr_matrix&       R        = hierarchy.restrictions[level];

        // A_fine.print_matrix("A_fine");

        connections.resize(A_fine.get_nnz(), 0);
        aggregates.resize(A_fine.get_m(), 0);

        std::cout << "1111" << std::endl;

        // Compute strength of connections
        compute_strong_connections(A_fine, eps, connections);

        std::cout << "2222" << std::endl;

        // Compute aggregations using parallel maximal independent set
        compute_aggregates_using_pmis(A_fine, connections, aggregates, aggregate_root_nodes);

        std::cout << "3333" << std::endl;

        // Construct prolongation matrix using smoothed aggregation
        construct_prolongation_using_unsmoothed_aggregation(
            A_fine, connections, aggregates, aggregate_root_nodes, P);

        std::cout << "4444" << std::endl;

        if(P.get_n() == 0)
        {
            break;
        }

        // Compute restriction matrix by transpose of prolongation matrix
        P.transpose(R);

        // Compute coarse grid matrix using Galarkin triple product A_c = R * A_f * P
        galarkin_triple_product(R, A_fine, P, A_coarse);

        level++;
        eps *= 0.5;
    }

    hierarchy.total_levels = level;

    std::cout << "Total number of levels in operator hierarchy at the end of the "
                 "setup phase: "
              << level << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Unsmoothed Aggregation amg setup time: " << ms_double.count() << "ms"
              << std::endl;
}