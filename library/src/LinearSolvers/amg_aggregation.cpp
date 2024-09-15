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
#include <assert.h>
#include "../../include/LinearSolvers/amg.h"
#include "../../include/LinearSolvers/amg_aggregation.h"
#include "../../include/LinearSolvers/slaf.h"

//********************************************************************************
//
// AMG: Smoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

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

static pmis_node lexographical_max(const pmis_node* ti, const pmis_node* tj)
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
		else if (state[i] == -2)
		{
			aggregates[i] = -2;
		}
	}
}

bool compute_aggregates_using_pmis(const csr_matrix& A, const std::vector<int>& connections, std::vector<int64_t>& aggregates, std::vector<int64_t>& aggregate_root_nodes)
{
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
		int64_t temp = aggregates[i];
		aggregates[i] = sum;
		sum += temp;
	}

	/*std::cout << "max_state" << std::endl;
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
	std::cout << "" << std::endl;*/

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

	std::cout << "aggregate_root_nodes final" << std::endl;
	for (size_t i = 0; i < aggregate_root_nodes.size(); i++)
	{
		std::cout << aggregate_root_nodes[i] << " ";
	}
	std::cout << "" << std::endl;

	return true;
}
