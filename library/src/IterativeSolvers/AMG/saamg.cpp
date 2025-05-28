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

#include "../../../include/IterativeSolvers/AMG/saamg.h"
#include "../../../include/IterativeSolvers/AMG/amg_aggregation.h"
#include "../../../include/IterativeSolvers/AMG/amg_strength.h"
#include "../../../include/IterativeSolvers/AMG/amg_util.h"
#include "../../../include/slaf.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
#include <chrono>

#include "../../trace.h"

//********************************************************************************
//
// AMG: Smoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

static bool construct_prolongation_using_smoothed_aggregation(const csr_matrix &A, const std::vector<int> &connections,
                                                              const std::vector<int64_t> &aggregates,
                                                              const std::vector<int64_t> &aggregate_root_nodes,
                                                              double relax, csr_matrix &prolongation)
{
    ROUTINE_TRACE("construct_prolongation_using_smoothed_aggregation");

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

    // Determine number of non-zeros for P
    for (int i = 0; i < A.m; i++)
    {
        int start = A.csr_row_ptr[i];
        int end = A.csr_row_ptr[i + 1];

        std::unordered_set<int64_t> table;

        for (int j = start; j < end; j++)
        {
            int col = A.csr_col_ind[j];

            // if diagonal entry or a strong connection, it contributes to
            // prolongation
            if (col == i || connections[j] == 1)
            {
                int64_t aggregate = aggregates[col];

                if (aggregate >= 0)
                {
                    table.insert(aggregate);
                }
            }
        }

        prolongation.csr_row_ptr[i + 1] = table.size();
        prolongation.nnz += table.size();
    }

    // exclusive scan on prolongation row pointer array
    prolongation.csr_row_ptr[0] = 0;
    for (int i = 0; i < prolongation.m; i++)
    {
        prolongation.csr_row_ptr[i + 1] += prolongation.csr_row_ptr[i];
    }

    // std::cout << "prolongation.csr_row_ptr" << std::endl;
    // for (size_t i = 0; i < prolongation.csr_row_ptr.size(); i++)
    // {
    // 	std::cout << prolongation.csr_row_ptr[i] << " ";
    // }
    // std::cout << "" << std::endl;

    assert(prolongation.nnz == prolongation.csr_row_ptr[prolongation.m]);

    prolongation.csr_col_ind.resize(prolongation.nnz);
    prolongation.csr_val.resize(prolongation.nnz);

    // Fill P
    for (int i = 0; i < A.m; i++)
    {
        std::map<int, double> table;

        int start = A.csr_row_ptr[i];
        int end = A.csr_row_ptr[i + 1];

        double diagonal = 0.0;

        // Diagonal of prolongation matrix is the diagonal of the original matrix
        // minus the original matrix weak connections
        for (int j = start; j < end; j++)
        {
            int col = A.csr_col_ind[j];

            if (col == i)
            {
                diagonal += A.csr_val[j];
            }
            else if (connections[j] == 0) // substract weak connections
            {
                diagonal -= A.csr_val[j];
            }
        }

        double invDiagonal = 1.0 / diagonal;

        for (int j = start; j < end; j++)
        {
            int col = A.csr_col_ind[j];

            // if diagonal entry or a strong connection, it contributes to
            // prolongation
            if (col == i || connections[j] == 1)
            {
                int aggregate = aggregates[col];

                if (aggregate >= 0)
                {
                    double value = (col == i) ? 1.0 - relax : -relax * invDiagonal * A.csr_val[j];

                    if (table.find(aggregate) != table.end())
                    {
                        table[aggregate] += value;
                    }
                    else
                    {
                        table[aggregate] = value;
                    }
                }
            }
        }

        int prolongation_start = prolongation.csr_row_ptr[i];

        int count = 0;
        for (auto it = table.begin(); it != table.end(); it++)
        {
            prolongation.csr_col_ind[prolongation_start + count] = it->first;
            prolongation.csr_val[prolongation_start + count] = it->second;
            ++count;
        }
    }

    return true;
}




















static bool construct_prolongation_using_smoothed_aggregation(const csr_matrix2 &A, const std::vector<int> &connections,
                                                              const std::vector<int64_t> &aggregates,
                                                              const std::vector<int64_t> &aggregate_root_nodes,
                                                              double relax, csr_matrix2 &prolongation)
{
    ROUTINE_TRACE("construct_prolongation_using_smoothed_aggregation");

    //prolongation.m = A.m;
    //prolongation.nnz = 0;
    //prolongation.csr_row_ptr.resize(A.m + 1, 0);
    //
    //// Determine number of columns in the prolongation matrix. This will be
    //// the maximum aggregate plus one.
    //prolongation.n = -1;
    //for (size_t i = 0; i < aggregates.size(); i++)
    //{
    //    if (prolongation.n < aggregates[i])
    //    {
    //        prolongation.n = aggregates[i];
    //    }
    //}

    //prolongation.n++;
    

    // Determine number of columns in the prolongation matrix. This will be
    // the maximum aggregate plus one.
    int n = -1;
    for (size_t i = 0; i < aggregates.size(); i++)
    {
       if (n < aggregates[i])
       {
           n = aggregates[i];
       }
    }
    n++;

    prolongation.resize(A.get_m(), n, 0);

    std::cout << "prolongation.n: " << prolongation.get_n() << " A.m: " << A.get_m() << "A.nnz: " << A.get_nnz() << std::endl;

    const int* csr_row_ptr_A = A.get_row_ptr();
    const int* csr_col_ind_A = A.get_col_ind();
    const double* csr_val_A = A.get_val();

    int* csr_row_ptr_P = prolongation.get_row_ptr();

    // Determine number of non-zeros for P
    int nnz = 0;
    for (int i = 0; i < A.get_m(); i++)
    {
        int start = csr_row_ptr_A[i];
        int end = csr_row_ptr_A[i + 1];

        std::unordered_set<int64_t> table;

        for (int j = start; j < end; j++)
        {
            int col = csr_col_ind_A[j];

            // if diagonal entry or a strong connection, it contributes to
            // prolongation
            if (col == i || connections[j] == 1)
            {
                int64_t aggregate = aggregates[col];

                if (aggregate >= 0)
                {
                    table.insert(aggregate);
                }
            }
        }

        csr_row_ptr_P[i + 1] = table.size();
        nnz += table.size();
    }

    // exclusive scan on prolongation row pointer array
    csr_row_ptr_P[0] = 0;
    for (int i = 0; i < prolongation.get_m(); i++)
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

    //prolongation.csr_col_ind.resize(prolongation.nnz);
    //prolongation.csr_val.resize(prolongation.nnz);
    prolongation.resize(prolongation.get_m(), prolongation.get_n(), nnz);

    // Fill P
    for (int i = 0; i < A.get_m(); i++)
    {
        std::map<int, double> table;

        int start = csr_row_ptr_A[i];
        int end = csr_row_ptr_A[i + 1];

        double diagonal = 0.0;

        // Diagonal of prolongation matrix is the diagonal of the original matrix
        // minus the original matrix weak connections
        for (int j = start; j < end; j++)
        {
            int col = csr_col_ind_A[j];

            if (col == i)
            {
                diagonal += csr_val_A[j];
            }
            else if (connections[j] == 0) // substract weak connections
            {
                diagonal -= csr_val_A[j];
            }
        }

        double invDiagonal = 1.0 / diagonal;

        for (int j = start; j < end; j++)
        {
            int col = csr_col_ind_A[j];

            // if diagonal entry or a strong connection, it contributes to
            // prolongation
            if (col == i || connections[j] == 1)
            {
                int aggregate = aggregates[col];

                if (aggregate >= 0)
                {
                    double value = (col == i) ? 1.0 - relax : -relax * invDiagonal * csr_val_A[j];

                    if (table.find(aggregate) != table.end())
                    {
                        table[aggregate] += value;
                    }
                    else
                    {
                        table[aggregate] = value;
                    }
                }
            }
        }

        int* csr_col_ind_P = prolongation.get_col_ind();
        double* csr_val_P = prolongation.get_val();

        int prolongation_start = csr_row_ptr_P[i];

        int count = 0;
        for (auto it = table.begin(); it != table.end(); it++)
        {
            csr_col_ind_P[prolongation_start + count] = it->first;
            csr_val_P[prolongation_start + count] = it->second;
            ++count;
        }
    }

    return true;
}



























void saamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
                 int max_level, heirarchy &hierarchy)
{
    ROUTINE_TRACE("saamg_setup");

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

    double eps = 0.08; // strength_of_connection/coupling strength
    double relax = 2.0 / 3.0;

    int level = 0;
    while (level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;
        std::cout << "2222" << std::endl;

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
        construct_prolongation_using_smoothed_aggregation(A, connections, aggregates, aggregate_root_nodes, relax, P);

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
    std::cout << "Smoothed Aggregation AMG setup time: " << ms_double.count() << "ms" << std::endl;
}

void saamg_setup(const csr_matrix2& mat_A, int max_level, heirarchy2 &hierarchy)
{
    ROUTINE_TRACE("saamg_setup");
    std::cout << "saamg_setup" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    hierarchy.prolongations.resize(max_level);
    hierarchy.restrictions.resize(max_level);
    hierarchy.A_cs.resize(max_level + 1);

    // Set original matrix at level 0 in the hierarchy
    hierarchy.A_cs[0].copy_from(mat_A);

    //hierarchy.A_cs[0].m = m;
    //hierarchy.A_cs[0].n = m;
    //hierarchy.A_cs[0].nnz = nnz;
    //hierarchy.A_cs[0].csr_row_ptr.resize(m + 1);
    //hierarchy.A_cs[0].csr_col_ind.resize(nnz);
    //hierarchy.A_cs[0].csr_val.resize(nnz);

    //for (int i = 0; i < m + 1; i++)
    //{
    //    hierarchy.A_cs[0].csr_row_ptr[i] = csr_row_ptr[i];
    //}

    //for (int i = 0; i < nnz; i++)
    //{
    //    hierarchy.A_cs[0].csr_col_ind[i] = csr_col_ind[i];
    //    hierarchy.A_cs[0].csr_val[i] = csr_val[i];
    //}

    double eps = 0.08; // strength_of_connection/coupling strength
    double relax = 2.0 / 3.0;

    int level = 0;
    while (level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;

        csr_matrix2 &A = hierarchy.A_cs[level];
        csr_matrix2 &A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix2 &P = hierarchy.prolongations[level];
        csr_matrix2 &R = hierarchy.restrictions[level];

        std::vector<int> connections;
        std::vector<int64_t> aggregates;
        std::vector<int64_t> aggregate_root_nodes;

        connections.resize(A.get_nnz(), 0);
        aggregates.resize(A.get_m(), 0);

        // Compute strength of connections
        compute_strong_connections(A, eps, connections);

        // Compute aggregations using parallel maximal independent set
        compute_aggregates_using_pmis(A, connections, aggregates, aggregate_root_nodes);

        // Construct prolongation matrix using smoothed aggregation
        construct_prolongation_using_smoothed_aggregation(A, connections, aggregates, aggregate_root_nodes, relax, P);

        if (P.get_n() == 0)
        {
            break;
        }

        // Compute restriction matrix by transpose of prolongation matrix
        //transpose(P, R);
        P.transpose(R);

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
    std::cout << "Smoothed Aggregation AMG setup time: " << ms_double.count() << "ms" << std::endl;
}