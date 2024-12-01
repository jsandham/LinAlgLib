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
// The above copyright noticeand this permission notice shall be included in all
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

#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <unordered_set>
#include <map>
#include "../../../include/LinearSolvers/AMG/amg_strength.h"
#include "../../../include/LinearSolvers/AMG/amg_aggregation.h"
#include "../../../include/LinearSolvers/AMG/saamg.h"
#include "../../../include/LinearSolvers/slaf.h"

//********************************************************************************
//
// AMG: Smoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

static bool construct_prolongation_using_smoothed_aggregation(const csr_matrix& A,
	const std::vector<int>& connections,
	const std::vector<int64_t>& aggregates,
	const std::vector<int64_t>& aggregate_root_nodes,
	double relax,
	csr_matrix& prolongation)
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

	//std::cout << "prolongation.n: " << prolongation.n << " A.m: " << A.m << " A.nnz: " << A.nnz << std::endl;

	// Determine number of non-zeros for P
	for (int i = 0; i < A.m; i++)
	{
		int start = A.csr_row_ptr[i];
		int end = A.csr_row_ptr[i + 1];

		std::unordered_set<int> table;

		for (int j = start; j < end; j++)
		{
			int col = A.csr_col_ind[j];

			// if diagonal entry or a strong connection, it contributes to prolongation
			if (col == i || connections[j] == 1)
			{
				int aggregate = aggregates[col];

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

		// Diagonal of prolongation matrix is the diagonal of the original matrix minus the original matrix weak connections
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

			// if diagonal entry or a strong connection, it contributes to prolongation
			if (col == i || connections[j] == 1)
			{
				int aggregate = aggregates[col];

				if (aggregate >= 0)
				{
					double value = (col == i) ? 1.0 - relax : -relax * invDiagonal * A.csr_val[j];
				
					if(table.find(aggregate) != table.end())
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
		for(auto it = table.begin(); it != table.end(); it++)
		{
			prolongation.csr_col_ind[prolongation_start + count] = it->first;
			prolongation.csr_val[prolongation_start + count] = it->second;
			++count;
		}
	}

	return true;
}

static void transpose(const csr_matrix& prolongation, csr_matrix& restriction)
{
	restriction.m = prolongation.n;
	restriction.n = prolongation.m;
	restriction.nnz = prolongation.nnz;
	restriction.csr_row_ptr.resize(restriction.m + 1);
	restriction.csr_col_ind.resize(restriction.nnz);
	restriction.csr_val.resize(restriction.nnz);

	// Fill arrays
	for(size_t i = 0; i < restriction.csr_row_ptr.size(); i++)
	{
		restriction.csr_row_ptr[i] = 0;
	}

	for(size_t i = 0; i < restriction.csr_col_ind.size(); i++)
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

// Compute C = alpha * A * B + beta * D
static void csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha,
	const int* csr_row_ptr_A, const int* csr_col_ind_A, const int* csr_row_ptr_B, const int* csr_col_ind_B,
	double beta, const int* csr_row_ptr_D, const int* csr_col_ind_D, int* csr_row_ptr_C, int* nnz_C)
{
	std::vector<int> nnz(n, -1);

	// A is mxk, B is kxn, and C is mxn
	for (int i = 0; i < m + 1; i++)
	{
		csr_row_ptr_C[i] = 0;
	}

	for (int i = 0; i < m; ++i)
	{
		int row_begin_A = csr_row_ptr_A[i];
		int row_end_A = csr_row_ptr_A[i + 1];

		for (int j = row_begin_A; j < row_end_A; j++)
		{
			int col_A = csr_col_ind_A[j];

			int row_begin_B = csr_row_ptr_B[col_A];
			int row_end_B = csr_row_ptr_B[col_A + 1];

			for (int p = row_begin_B; p < row_end_B; p++)
			{
				int col_B = csr_col_ind_B[p];

				if (nnz[col_B] != i)
				{
					nnz[col_B] = i;
					csr_row_ptr_C[i + 1]++;
				}
			}
		}

		if (beta != 0.0)
		{
			int row_begin_D = csr_row_ptr_D[i];
			int row_end_D = csr_row_ptr_D[i + 1];

			for (int j = row_begin_D; j < row_end_D; j++)
			{
				int col_D = csr_col_ind_D[j];

				if (nnz[col_D] != i)
				{
					nnz[col_D] = i;
					csr_row_ptr_C[i + 1]++;
				}
			}
		}
	}

	for (int i = 0; i < m; i++)
	{
		csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
	}

	*nnz_C = csr_row_ptr_C[m];
}

static void csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha,
	const int* csr_row_ptr_A, const int* csr_col_ind_A, const double* csr_val_A,
	const int* csr_row_ptr_B, const int* csr_col_ind_B, const double* csr_val_B,
	double beta, const int* csr_row_ptr_D, const int* csr_col_ind_D, const double* csr_val_D,
	const int* csr_row_ptr_C, int* csr_col_ind_C, double* csr_val_C)
{
	std::vector<int> nnzs(n, -1);

	for (int i = 0; i < m; i++)
	{
		int row_begin_C = csr_row_ptr_C[i];
		int row_end_C = row_begin_C;

		int row_begin_A = csr_row_ptr_A[i];
		int row_end_A = csr_row_ptr_A[i + 1];

		for (int j = row_begin_A; j < row_end_A; j++)
		{
			int col_A = csr_col_ind_A[j];
			double val_A = alpha * csr_val_A[j];

			int row_begin_B = csr_row_ptr_B[col_A];
			int row_end_B = csr_row_ptr_B[col_A + 1];

			for (int p = row_begin_B; p < row_end_B; p++)
			{
				int col_B = csr_col_ind_B[p];
				double val_B = csr_val_B[p];

				if (nnzs[col_B] < row_begin_C)
				{
					nnzs[col_B] = row_end_C;
					csr_col_ind_C[row_end_C] = col_B;
					csr_val_C[row_end_C] = val_A * val_B;
					row_end_C++;
				}
				else
				{
					csr_val_C[nnzs[col_B]] += val_A * val_B;
				}
			}
		}

		if (beta != 0.0)
		{
			int row_begin_D = csr_row_ptr_D[i];
			int row_end_D = csr_row_ptr_D[i + 1];

			for (int j = row_begin_D; j < row_end_D; j++)
			{
				int col_D = csr_col_ind_D[j];
				double val_D = beta * csr_val_D[j];

				// Check if a new nnz is generated or if the value is added
				if (nnzs[col_D] < row_begin_C)
				{
					nnzs[col_D] = row_end_C;

					csr_col_ind_C[row_end_C] = col_D;
					csr_val_C[row_end_C] = val_D;
					row_end_C++;
				}
				else
				{
					csr_val_C[nnzs[col_D]] += val_D;
				}
			}
		}
	}

	int nnz = csr_row_ptr_C[m];

	std::vector<int> cols(nnz);
	std::vector<double> vals(nnz);

	memcpy(cols.data(), csr_col_ind_C, sizeof(int) * nnz);
	memcpy(vals.data(), csr_val_C, sizeof(double) * nnz);

	for (int i = 0; i < m; i++)
	{
		int row_begin = csr_row_ptr_C[i];
		int row_end = csr_row_ptr_C[i + 1];
		int row_nnz = row_end - row_begin;

		std::vector<int> perm(row_nnz);
		for (int j = 0; j < row_nnz; j++)
		{
			perm[j] = j;
		}

		int* col_entry = cols.data() + row_begin;
		double* val_entry = vals.data() + row_begin;

		std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
			return col_entry[a] < col_entry[b];
			});

		for (int j = 0; j < row_nnz; j++)
		{
			csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
			csr_val_C[row_begin + j] = val_entry[perm[j]];
		}
	}
}

static void galarkinTripleProduct(const csr_matrix& R, const csr_matrix& A, const csr_matrix& P, csr_matrix& A_coarse)
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

	csrgemm_nnz(A.m, P.n, A.n, A.nnz, P.nnz, 0, alpha, A.csr_row_ptr.data(), A.csr_col_ind.data(),
		P.csr_row_ptr.data(), P.csr_col_ind.data(), beta, nullptr, nullptr, AP.csr_row_ptr.data(), &AP.nnz);

	//std::cout << "AP.nnz: " << AP.nnz << std::endl;
	AP.csr_col_ind.resize(AP.nnz);
	AP.csr_val.resize(AP.nnz);

	csrgemm(A.m, P.n, A.n, A.nnz, P.nnz, 0, alpha,
		A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(),
		P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), beta,
		nullptr, nullptr, nullptr, AP.csr_row_ptr.data(), AP.csr_col_ind.data(), AP.csr_val.data());

	// Determine number of non-zeros in A_coarse = R * AP product
	A_coarse.m = R.m;
	A_coarse.n = AP.n;
	A_coarse.nnz = 0;
	A_coarse.csr_row_ptr.resize(A_coarse.m + 1, 0);

	csrgemm_nnz(R.m, AP.n, R.n, R.nnz, AP.nnz, 0, alpha, R.csr_row_ptr.data(), R.csr_col_ind.data(),
		AP.csr_row_ptr.data(), AP.csr_col_ind.data(), beta, nullptr, nullptr, A_coarse.csr_row_ptr.data(), &A_coarse.nnz);

	//std::cout << "A_coarse.nnz: " << A_coarse.nnz << std::endl;
	A_coarse.csr_col_ind.resize(A_coarse.nnz);
	A_coarse.csr_val.resize(A_coarse.nnz);

	csrgemm(R.m, AP.n, R.n, R.nnz, AP.nnz, 0, alpha,
		R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(),
		AP.csr_row_ptr.data(), AP.csr_col_ind.data(), AP.csr_val.data(), beta,
		nullptr, nullptr, nullptr, A_coarse.csr_row_ptr.data(), A_coarse.csr_col_ind.data(), A_coarse.csr_val.data());



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

void saamg_setup(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, int m, int n, int nnz, int max_level, heirarchy& hierarchy)
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

	double eps = 0.08; // strength_of_connection/coupling strength
	double relax = 2.0 / 3.0;

	int level = 0;
	while (level < max_level)
	{
		std::cout << "Compute operators at coarse level: " << level << std::endl;
		
		csr_matrix& A = hierarchy.A_cs[level];
		csr_matrix& A_coarse = hierarchy.A_cs[level + 1];
		csr_matrix& P = hierarchy.prolongations[level];
		csr_matrix& R = hierarchy.restrictions[level];

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
		galarkinTripleProduct(R, A, P, A_coarse);

		level++;
		eps *= 0.5;
	}

	hierarchy.total_levels = level;

	std::cout << "Total number of levels in operator hierarchy at the end of the setup phase: " << level << std::endl;
}
