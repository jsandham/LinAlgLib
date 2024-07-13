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
#include <stdlib.h>
#include "../../include/LinearSolvers/saamg.h"
#include "../../include/LinearSolvers/slaf.h"
#include "math.h"

//********************************************************************************
//
// AMG: Smoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

#define DEBUG 1
#define FAST_ERROR 0
#define MAX_VCYCLES 100

static void extract_diaganal(const csr_matrix& A, std::vector<double>& diag)
{
	assert(A.m == diag.size());

	for (int i = 0; i < A.m; i++)
	{
		int row_start = A.csr_row_ptr[i];
		int row_end = A.csr_row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++)
		{
			if (A.csr_col_ind[j] == i)
			{
				diag[i] = A.csr_val[j];
				break;
			}
		}
	}
}

static void compute_strong_connections(const csr_matrix& A, const std::vector<double>& diag, std::vector<int>& connections)
{
	double eps2 = 0.1;

	for (int i = 0; i < A.m; i++)
	{
		double eps_dia_i = eps2 * diag[i];

		int row_start = A.csr_row_ptr[i];
		int row_end = A.csr_row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++)
		{
			int       c = A.csr_col_ind[j];
			double v = A.csr_val[j];

			assert(c >= 0);
			assert(c < A.m);

			connections[j] = (c != i) && (v * v > eps_dia_i * diag[c]);
		}
	}
}

static unsigned int hash1(unsigned int x)
{
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x / 2;
}

static void initialize_pmis_state(const csr_matrix& A, const std::vector<int>& connections, std::vector<int>& state, std::vector<int>& hash)
{
	for (int i = 0; i < A.m; ++i)
	{
		int s = -2;

		int row_start = A.csr_row_ptr[i];
		int row_end = A.csr_row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++)
		{
			if (connections[j] == 1)
			{
				s = 0;
				break;
			}
		}

		state[i] = s;
		hash[i] = hash1(i);
	}
}

struct pmis_node
{
	int state;
	int hash;
	int row;
};

pmis_node lexographical_max(pmis_node* ti, pmis_node* tj)
{
	// find lexographical maximum
	if (tj->state > ti->state)
	{
		return *tj;
	}
	else if (tj->state == ti->state)
	{
		if (tj->hash > ti->hash)
		{
			return *tj;
		}
	}

	return *ti;
}

static void find_maximum_distance_two_node(const csr_matrix& A, const std::vector<int>& connections, const std::vector<int>& state, 
	const std::vector<int>& hash, std::vector<int64_t>& aggregates, std::vector<int>& max_state, bool& complete)
{
	// Find distance 1 maximum neighbour node
	for (int i = 0; i < A.m; i++)
	{
		pmis_node max_node;
		max_node.state = state[i];
		max_node.hash = hash[i];
		max_node.row = i;

		int row_start = A.csr_row_ptr[i];
		int row_end = A.csr_row_ptr[i + 1];

		for (int j = row_start; j < row_end; j++)
		{
			if (connections[j] == 1)
			{
				int col = A.csr_col_ind[j];

				pmis_node node;
				node.state = state[col];
				node.hash = hash[col];
				node.row = col;

				max_node = lexographical_max(&max_node, &node);
			}
		}

		// Find distance 2 maximum neighbour node
		int row_start2 = A.csr_row_ptr[max_node.row];
		int row_end2 = A.csr_row_ptr[max_node.row + 1];
		
		for (int j = row_start2; j < row_end2; j++)
		{
			if (connections[j] == 1)
			{
				int col = A.csr_col_ind[j];

				pmis_node node;
				node.state = state[col];
				node.hash = hash[col];
				node.row = col;

				max_node = lexographical_max(&max_node, &node);
			}
		}

		if (state[i] == 0)
		{
			// If max node is current node, then make current node an aggregate root.
			if (max_node.row == i)
			{
				max_state[i] = 1;
				aggregates[i] = 1;
			}
			// If max node is not current node, but max node has a state of 1, then the max node must be an already existing aggregate root 
			// and therefore the current node is too close to an existing aggregate root for it to also be an aggregate root. We mark it 
			// with state -1 to indicate it cannot be an aggregate root.
			else if (max_node.state == 1)
			{
				max_state[i] = -1;
				aggregates[i] = 0;
			}
			// If max node is not current node, and also does not have a state of 1, then we must call this function again so we mark 
			// the work as not complete.
			else
			{
				complete = false;
			}
		}
	}
}

static void add_unassigned_nodes_to_closest_aggregation(const csr_matrix& A, const std::vector<int>& connections, const std::vector<int>& state,
	std::vector<int64_t>& aggregates, std::vector<int64_t>& aggregate_root_nodes, std::vector<int>& max_state)
{
	for (int i = 0; i < A.m; i++)
	{
		if (state[i] == -1)
		{
			int start = A.csr_row_ptr[i];
			int end = A.csr_row_ptr[i + 1];

			for (int j = start; j < end; j++)
			{
				if (connections[j] == 1)
				{
					int col = A.csr_col_ind[j];

					if (state[col] == 1)
					{
						aggregates[i] = aggregates[col];
						max_state[i] = 1;
						break;
					}
				}
			}
		}
		else if(state[i] == -2)
		{
			aggregates[i] = -2;
		}
	}
}

static bool compute_aggregates_using_pmis(const csr_matrix& A, std::vector<int>& connections, std::vector<int64_t>& aggregates, std::vector<int64_t>& aggregate_root_nodes)
{
	// Extract diagaonl
	std::vector<double> diag(A.m);
	extract_diaganal(A, diag);

	std::cout << "diag" << std::endl;
	for (size_t i = 0; i < diag.size(); i++)
	{
		std::cout << diag[i] << " ";
	}
	std::cout << "" << std::endl;

	std::cout << "A.nnz: " << A.nnz << " A.m: " << A.m << std::endl;

	connections.resize(A.nnz, 0);
	aggregates.resize(A.m, 0);

	// Compute connections
	compute_strong_connections(A, diag, connections);

	std::cout << "connections" << std::endl;
	for (size_t i= 0; i < connections.size(); i++)
	{
		std::cout << connections[i] << " ";
	}
	std::cout << "" << std::endl;

	std::vector<int> hash(A.m);
	std::vector<int> state(A.m);
	std::vector<int> max_state(A.m);

	// Initialize parallel maximal independent set state
	initialize_pmis_state(A, connections, max_state, hash);

	std::cout << "max_state" << std::endl;
	for (size_t i = 0; i < max_state.size(); i++)
	{
		std::cout << max_state[i] << " ";
	}
	std::cout << "" << std::endl;

	std::cout << "hash" << std::endl;
	for (size_t i = 0; i < hash.size(); i++)
	{
		std::cout << hash[i] << " ";
	}
	std::cout << "" << std::endl;

	int iter = 0;
	while (iter < 20)
	{
		for (int i = 0; i < A.m; i++)
		{
			state[i] = max_state[i];
		}

		// Find maximum distance 2 node
		bool complete = true;
		find_maximum_distance_two_node(A, connections, state, hash, aggregates, max_state, complete);

		if (complete)
		{
			break;
		}

		if (iter > 20)
		{
			std::cout << "Hit maximum iterations when determinig aggregates" << std::endl;
			break;
		}

		iter++;
	}

	aggregate_root_nodes.resize(A.m, -1);

	for (size_t i = 0; i < aggregates.size(); i++)
	{
		aggregate_root_nodes[i] = (aggregates[i] == 1) ? 1 : -1;
	}

	std::cout << "aggregates before exclusive sum" << std::endl;
	for (size_t i = 0; i < aggregates.size(); i++)
	{
		std::cout << aggregates[i] << " ";
	}
	std::cout << "" << std::endl;

	// 1 0 0 1 1 1
	// 0 1 1 1 2 3

	// Exclusive sum
	int64_t sum = 0;
	for (int i = 0; i < A.m; i++)
	{
		int temp = aggregates[i];
		aggregates[i] = sum;
		sum += temp;
	}

	std::cout << "max_state" << std::endl;
	for (size_t i = 0; i < max_state.size(); i++)
	{
		std::cout << max_state[i] << " ";
	}
	std::cout << "" << std::endl;

	std::cout << "aggregates after exclusive sum" << std::endl;
	for (size_t i = 0; i < aggregates.size(); i++)
	{
		std::cout << aggregates[i] << " ";
	}
	std::cout << "" << std::endl;

	std::cout << "aggregate_root_nodes" << std::endl;
	for (size_t i = 0; i < aggregate_root_nodes.size(); i++)
	{
		std::cout << aggregate_root_nodes[i] << " ";
	}
	std::cout << "" << std::endl;

	// Add any unassigned nodes to an existing aggregation
	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < A.m; i++)
		{
			state[i] = max_state[i];
		}

		add_unassigned_nodes_to_closest_aggregation(A, connections, state, aggregates, aggregate_root_nodes, max_state);
	}

	std::cout << "aggregates final" << std::endl;
	for (size_t i = 0; i < aggregates.size(); i++)
	{
		std::cout << aggregates[i] << " ";
	}
	std::cout << "" << std::endl;

	return true;
}

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

	std::cout << "prolongation.n: " << prolongation.n << " A.m: " << A.m << " A.nnz: " << A.nnz << std::endl;

	std::vector<int> table(prolongation.n, -1);

	// Determine number of non-zeros for P
	for (int i = 0; i < A.m; i++)
	{
		int start = A.csr_row_ptr[i];
		int end = A.csr_row_ptr[i + 1];

		for (int j = start; j < end; j++)
		{
			int col = A.csr_col_ind[j];

			// if diagonal entry or a strong connection, it contributes to prolongation
			if (col == i || connections[j] == 1)
			{
				int aggregate = aggregates[col];

				if (aggregate >= 0 && table[aggregate] != i)
				{
					prolongation.nnz++;
					prolongation.csr_row_ptr[i + 1]++;

					table[aggregate] = i;
				}
			}
		}
	}

	// exclusive scan on prolongation row pointer array
	prolongation.csr_row_ptr[0] = 0;
	for (int i = 0; i < prolongation.m; i++)
	{
		prolongation.csr_row_ptr[i + 1] += prolongation.csr_row_ptr[i];
	}

	std::cout << "prolongation.csr_row_ptr" << std::endl;
	for (size_t i = 0; i < prolongation.csr_row_ptr.size(); i++)
	{
		std::cout << prolongation.csr_row_ptr[i] << " ";
	}
	std::cout << "" << std::endl;

	assert(prolongation.nnz == prolongation.csr_row_ptr[prolongation.m]);

	prolongation.csr_col_ind.resize(prolongation.nnz);
	prolongation.csr_val.resize(prolongation.nnz);

	// reset table to -1
	for (size_t i = 0; i < table.size(); i++)
	{
		table[i] = -1;
	}

	// Fill P
	for (int i = 0; i < A.m; i++)
	{
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
			else if (connections[j] == 0)
			{
				diagonal -= A.csr_val[j];
			}
		}

		double invDiagonal = 1.0 / diagonal;

		int prolongation_start = prolongation.csr_row_ptr[i];
		int prolongation_end = prolongation_start;

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
				
					if (table[aggregate] < prolongation_start)
					{
						prolongation.csr_col_ind[prolongation_end] = aggregate;
						prolongation.csr_val[prolongation_end] = value;
						
						table[aggregate] = prolongation_end;
						prolongation_end++;
					}
					else
					{
						prolongation.csr_val[table[aggregate]] += value;
					}
				}
			}
		}

		assert(prolongation_start == prolongation.csr_row_ptr[i]);
		assert(prolongation_end == prolongation.csr_row_ptr[i + 1]);

		int prolongation_row_nnz = prolongation_end - prolongation_start;

		// sort columns
		std::vector<int> perm(prolongation_row_nnz);
		for (int j = 0; j < prolongation_row_nnz; j++)
		{
			perm[j] = j;
		}

		int* col_entry = prolongation.csr_col_ind.data() + prolongation_start;
		double* val_entry = prolongation.csr_val.data() + prolongation_start;

		std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
			return col_entry[a] < col_entry[b];
			});

		for (int j = 0; j < prolongation_row_nnz; j++)
		{
			prolongation.csr_col_ind[prolongation_start + j] = col_entry[perm[j]];
			prolongation.csr_val[prolongation_start + j] = val_entry[perm[j]];
		}
	}

	return true;
}

static void transpose(const csr_matrix& prolongation, csr_matrix& restriction)
{
	restriction.m = prolongation.n;
	restriction.n = prolongation.m;
	restriction.nnz = prolongation.nnz;
	restriction.csr_row_ptr.resize(restriction.m + 1, 0);
	restriction.csr_col_ind.resize(restriction.nnz, -1);
	restriction.csr_val.resize(restriction.nnz);

	std::cout << "prolongation" << std::endl;
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
	std::cout << "" << std::endl;

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

	std::cout << "restriction" << std::endl;
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
	std::cout << "" << std::endl;
}

// Compute y = alpha * A * x + beta * y
static void csrmv(int m, int n, int nnz, double alpha, const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val,
	const double* x, double beta, double* y)
{
	for (int i = 0; i < m; i++)
	{
		int row_begin = csr_row_ptr[i];
		int row_end = csr_row_ptr[i + 1];

		double sum = 0;
		for (int j = row_begin; j < row_end; j++)
		{
			sum += csr_val[j] * x[csr_col_ind[j]];
		}

		y[i] = alpha * sum + beta * y[i];
	}
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

	std::cout << "AP.nnz: " << AP.nnz << std::endl;
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

	std::cout << "A_coarse.nnz: " << A_coarse.nnz << std::endl;
	A_coarse.csr_col_ind.resize(A_coarse.nnz);
	A_coarse.csr_val.resize(A_coarse.nnz);

	csrgemm(R.m, AP.n, R.n, R.nnz, AP.nnz, 0, alpha,
		R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(),
		AP.csr_row_ptr.data(), AP.csr_col_ind.data(), AP.csr_val.data(), beta,
		nullptr, nullptr, nullptr, A_coarse.csr_row_ptr.data(), A_coarse.csr_col_ind.data(), A_coarse.csr_val.data());



	std::cout << "A coarse" << std::endl;
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
	std::cout << "" << std::endl;
}

void saamg_setup(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, int m, int n, int nnz, int max_level, double theta, saamg_heirarchy& hierarchy)
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

	double relax = 0.001;

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
		relax *= 0.5;
	}

	std::cout << "Total number of levels in operator hierarchy at the end of the setup phase: " << level << std::endl;
}

void gauss_siedel_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, const int n);

static void saamg_vcycle(const saamg_heirarchy& hierarchy, double* x, const double* b, int n1, int n2, const double theta, int currentLevel)
{
	constexpr int MAX_LEVELS = 40;

	// A_coarse = R*A*P
	const csr_matrix& A = hierarchy.A_cs[currentLevel];
	const csr_matrix& restriction = hierarchy.restrictions[currentLevel];
	const csr_matrix& prolongation = hierarchy.prolongations[currentLevel];

	int N = A.m; // size of A at current level

	if (currentLevel < MAX_LEVELS) 
	{
		const csr_matrix& A_coarse = hierarchy.A_cs[currentLevel + 1];

		int Nc = A_coarse.m; //size of A at next course level

		// do n1 smoothing steps on A*x=b
		for (int i = 0; i < n1; i++) 
		{
			// Gauss-Seidel iteration
			gauss_siedel_iteration(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N);
		}

		// compute residual r = b - A*x = A*e
		std::vector<double> r(N);
		for (int i = 0; i < N; i++) 
		{
			double Ax = 0.0;  //matrix vector product Ax

			int row_start = A.csr_row_ptr[i];
			int row_end = A.csr_row_ptr[i + 1];

			for (int j = row_start; j < row_end; j++) 
			{
				Ax = Ax + A.csr_val[j] * x[A.csr_col_ind[j]];
			}
			r[i] = b[i] - Ax;
		}

		// compute wres = R*r 
		std::vector<double> wres(Nc, 0.0);
		for (int i = 0; i < N; i++) 
		{
			int row_start = restriction.csr_row_ptr[i];
			int row_end = restriction.csr_row_ptr[i + 1];

			for (int j = row_start; j < row_end; j++) 
			{
				wres[restriction.csr_col_ind[j]] = wres[restriction.csr_col_ind[j]] + restriction.csr_val[j] * r[i];
			}
		}

		// set e = 0
		std::vector<double> e(Nc, 0.0);

		// recursively solve Ac*ec = R*r = wres
		saamg_vcycle(hierarchy, e.data(), wres.data(), n1, n2, theta, currentLevel + 1);

		// correct x = x + P*ec
		for (int i = 0; i < N; i++) 
		{
			int row_start = prolongation.csr_row_ptr[i];
			int row_end = prolongation.csr_row_ptr[i + 1];

			for (int j = row_start; j < row_end; j++) 
			{
				x[i] = x[i] + prolongation.csr_val[j] * e[prolongation.csr_col_ind[j]];
			}
		}

		// do n2 smoothing steps on A*x=b
		for (int i = 0; i < n2; i++) 
		{
			// Gauss-Seidel iteration
			gauss_siedel_iteration(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N);
		}
	}
	else 
	{
		// solve A*x=b exactly
		for (int i = 0; i < 100; i++) 
		{
			// Gauss-Seidel iteration
			for (int j = 0; j < N; j++) 
			{
				double sigma = 0.0;
				double ajj = 0.0; // diagonal entry a_jj

				int row_start = A.csr_row_ptr[j];
				int row_end = A.csr_row_ptr[j + 1];

				for (int k = row_start; k < row_end; k++) 
				{
					int col = A.csr_col_ind[k];
					double val = A.csr_val[k];
					if (col != j)
					{
						sigma = sigma + val * x[col];
					}
					else
					{
						ajj = val;
					}
				}
				x[j] = (b[j] - sigma) / ajj;
			}
		}
	}
}

void saamg_solve(const saamg_heirarchy& hierarchy, double* x, const double* b, int n1, int n2, double theta, double tol)
{
	// AMG recursive solve
	int vcycle = 0;
	double err = 1.0;
	while (err > tol && vcycle < MAX_VCYCLES) 
	{
		saamg_vcycle(hierarchy, x, b, n1, n2, 0.5, 0);
#if(FAST_ERROR)
		err = fast_error(hierarchy.A_cs[0].csr_row_ptr.data(), hierarchy.A_cs[0].csr_col_ind.data(), hierarchy.A_cs[0].csr_val.data(), x, b, hierarchy.A_cs[0].m, tol);
#else
		err = error(hierarchy.A_cs[0].csr_row_ptr.data(), hierarchy.A_cs[0].csr_col_ind.data(), hierarchy.A_cs[0].csr_val.data(), x, b, hierarchy.A_cs[0].m);
#endif
#if(DEBUG)
		std::cout << "error: " << err << std::endl;
#endif
		vcycle++;
	}
}

