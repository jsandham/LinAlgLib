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
#include <assert.h>

#include "host_amg_aggregation.h"

#include "../../trace.h"

static unsigned int hash1(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x / 2;
}

void linalg::host_initialize_pmis_state(const csr_matrix&  A,
                                        const vector<int>& connections,
                                        vector<int>&       state,
                                        vector<int>&       hash)
{
    ROUTINE_TRACE("initialize_pmis_state");

    const int* csr_row_ptr_A = A.get_row_ptr();

    for(int i = 0; i < A.get_m(); ++i)
    {
        int s = -2;

        int row_start = csr_row_ptr_A[i];
        int row_end   = csr_row_ptr_A[i + 1];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                s = 0;
                break;
            }
        }

        state[i] = s;
        hash[i]  = hash1(i);
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
    if(tj->state > ti->state)
    {
        return *tj;
    }
    else if(tj->state == ti->state)
    {
        if(tj->hash > ti->hash)
        {
            return *tj;
        }
    }

    return *ti;
}

void linalg::host_find_maximum_distance_two_node(const csr_matrix&  A,
                                                 const vector<int>& connections,
                                                 const vector<int>& state,
                                                 const vector<int>& hash,
                                                 vector<int64_t>&   aggregates,
                                                 vector<int>&       max_state,
                                                 bool&              complete)
{
    ROUTINE_TRACE("find_maximum_distance_two_node");

    const int* csr_row_ptr_A = A.get_row_ptr();
    const int* csr_col_ind_A = A.get_col_ind();

    // Find distance 1 maximum neighbour node
    for(int i = 0; i < A.get_m(); i++)
    {
        pmis_node max_node;
        max_node.state = state[i];
        max_node.hash  = hash[i];
        max_node.row   = i;

        int row_start = csr_row_ptr_A[i];
        int row_end   = csr_row_ptr_A[i + 1];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                int col = csr_col_ind_A[j];

                pmis_node node;
                node.state = state[col];
                node.hash  = hash[col];
                node.row   = col;

                max_node = lexographical_max(&max_node, &node);
            }
        }

        // Find distance 2 maximum neighbour node
        int row_start2 = csr_row_ptr_A[max_node.row];
        int row_end2   = csr_row_ptr_A[max_node.row + 1];

        for(int j = row_start2; j < row_end2; j++)
        {
            if(connections[j] == 1)
            {
                int col = csr_col_ind_A[j];

                pmis_node node;
                node.state = state[col];
                node.hash  = hash[col];
                node.row   = col;

                max_node = lexographical_max(&max_node, &node);
            }
        }

        if(state[i] == 0)
        {
            // If max node is current node, then make current node an aggregate root.
            if(max_node.row == i)
            {
                max_state[i]  = 1;
                aggregates[i] = 1;
            }
            // If max node is not current node, but max node has a state of 1, then
            // the max node must be an already existing aggregate root and therefore
            // the current node is too close to an existing aggregate root for it to
            // also be an aggregate root. We mark it with state -1 to indicate it
            // cannot be an aggregate root.
            else if(max_node.state == 1)
            {
                max_state[i]  = -1;
                aggregates[i] = 0;
            }
            // If max node is not current node, and also does not have a state of 1,
            // then we must call this function again so we mark the work as not
            // complete.
            else
            {
                complete = false;
            }
        }
    }
}

void linalg::host_add_unassigned_nodes_to_closest_aggregation(const csr_matrix&  A,
                                                              const vector<int>& connections,
                                                              const vector<int>& state,
                                                              vector<int64_t>&   aggregates,
                                                              vector<int64_t>& aggregate_root_nodes,
                                                              vector<int>&     max_state)
{
    ROUTINE_TRACE("add_unassigned_nodes_to_closest_aggregation");

    const int* csr_row_ptr_A = A.get_row_ptr();
    const int* csr_col_ind_A = A.get_col_ind();

    for(int i = 0; i < A.get_m(); i++)
    {
        if(state[i] == -1)
        {
            int start = csr_row_ptr_A[i];
            int end   = csr_row_ptr_A[i + 1];

            for(int j = start; j < end; j++)
            {
                if(connections[j] == 1)
                {
                    int col = csr_col_ind_A[j];

                    if(state[col] == 1)
                    {
                        aggregates[i] = aggregates[col];
                        max_state[i]  = 1;
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

#define F_POINT 0 // F-point
#define C_POINT 1 // C-point
#define U_POINT 2 // Unassigned point
#define FF_POINT 3 // Future F-point

void linalg::host_compute_cfpoint_first_pass(const csr_matrix& S,
                                             const csr_matrix& ST,
                                             vector<uint32_t>& cfpoints)
{
    ROUTINE_TRACE("linalg::host_compute_cfpoint_first_pass");

    assert(S.get_m() == S.get_n());
    assert(ST.get_m() == ST.get_n());

    const int* csr_row_ptr_S = S.get_row_ptr();
    const int* csr_col_ind_S = S.get_col_ind();

    const int* csr_row_ptr_ST = ST.get_row_ptr();
    const int* csr_col_ind_ST = ST.get_col_ind();

    // Start by setting all points as unassigned
    for(int i = 0; i < S.get_m(); i++)
    {
        cfpoints[i] = U_POINT;
    }

    std::vector<int> lambda(ST.get_m(), 0);

    for(int i = 0; i < ST.get_m(); i++)
    {
        lambda[i] = csr_row_ptr_ST[i + 1] - csr_row_ptr_ST[i];
    }

    // lambda = [2, 0, 4, 4, 2, 0, 1, 2, 3, 2]
    // ptr = [0, 2, 1, 4, 1, 2, 0, 0, 0, 0, 0]
    // ptr = [0, 2, 3, 7, 8, 10, 10, 10, 10, 10, 10]
    // count = [2, 1, 4, 1, 2]
    // i2n = [1, 5, 6, 0, 4, 7, 9, 8, 2, 3]
    // n2i = [3, 0, 8, 9, 4, 1, 2, 5, 7, 6]
    std::vector<int> ptr(S.get_m() + 1, 0);
    std::vector<int> count(S.get_m(), 0);

    for(int i = 0; i < S.get_m(); i++)
    {
        ptr[lambda[i] + 1]++;
    }

    // Exclusive scan
    for(int i = 0; i < S.get_m(); i++)
    {
        ptr[i + 1] += ptr[i];
    }

    std::vector<int> i2n(S.get_m());
    std::vector<int> n2i(S.get_m());

    for(int i = 0; i < S.get_m(); i++)
    {
        int lam    = lambda[i];
        int index  = ptr[lam] + count[lam];
        i2n[index] = i;
        n2i[i]     = index;
        count[lam]++;
    }

    for(int i = S.get_m() - 1; i >= 0; i--)
    {
        int current_node = i2n[i];

        if(lambda[current_node] <= 0)
        {
            break;
        }

        if(cfpoints[current_node] == U_POINT)
        {
            // Node with most strong connections is made the first C-point
            cfpoints[current_node] = C_POINT;

            // Mark all neighbours of the C-point as future F-points
            int ST_row_start = csr_row_ptr_ST[current_node];
            int ST_row_end   = csr_row_ptr_ST[current_node + 1];

            for(int j = ST_row_start; j < ST_row_end; j++)
            {
                int ST_col_j = csr_col_ind_ST[j];

                if(cfpoints[ST_col_j] == U_POINT)
                {
                    cfpoints[ST_col_j] = FF_POINT;
                }
            }

            for(int j = ST_row_start; j < ST_row_end; j++)
            {
                int ST_col_j = csr_col_ind_ST[j];

                if(cfpoints[ST_col_j] == FF_POINT)
                {
                    cfpoints[ST_col_j] = F_POINT;

                    // Increment lambda value for all unassigned neighbours of F-point
                    int S_row_start = csr_row_ptr_S[ST_col_j];
                    int S_row_end   = csr_row_ptr_S[ST_col_j + 1];

                    for(int k = S_row_start; k < S_row_end; k++)
                    {
                        int S_col_k = csr_col_ind_S[k];

                        if(cfpoints[S_col_k] == U_POINT)
                        {
                            // lambda = [2, 0, 4, 4, 2, 0, 1, 2, 3, 2]
                            // ptr = [0, 2, 1, 4, 1, 2, 0, 0, 0, 0, 0]
                            // ptr = [0, 2, 3, 7, 8, 10, 10, 10, 10, 10, 10]
                            // count = [2, 1, 4, 1, 2]
                            // i2n = [1, 5, 6, 0, 4, 7, 9, 8, 2, 3]
                            // n2i = [3, 0, 8, 9, 4, 1, 2, 5, 7, 6]

                            // lambda = [2, 1, 4, 4, 2, 0, 1, 2, 3, 2]
                            // ptr = [0, 1, 2, 4, 1, 2, 0, 0, 0, 0, 0]
                            // ptr = [0, 1, 3, 7, 8, 10, 10, 10, 10, 10, 10]
                            // count = [1, 2, 4, 1, 2]
                            // i2n = [5, 1, 6, 0, 4, 7, 9, 8, 2, 3]
                            // n2i = [3, 1, 8, 9, 4, 0, 2, 5, 7, 6]

                            // I lambda_k = lambda[k];
                            // I old_pos  = node_to_index[k];
                            // I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

                            // node_to_index[index_to_node[old_pos]] = new_pos;
                            // node_to_index[index_to_node[new_pos]] = old_pos;
                            // std::swap(index_to_node[old_pos], index_to_node[new_pos]);

                            // // Update intervals
                            // interval_count[lambda_k]   -= 1;
                            // interval_count[lambda_k+1] += 1; //invalid write!
                            // interval_ptr[lambda_k+1]    = new_pos;

                            // // Increment lambda_k
                            // lambda[k]++;

                            int old_pos = n2i[S_col_k];
                            int new_pos = ptr[lambda[S_col_k]] + count[lambda[S_col_k]] - 1;

                            n2i[i2n[old_pos]] = new_pos;
                            n2i[i2n[new_pos]] = old_pos;

                            std::swap(i2n[old_pos], i2n[new_pos]);

                            count[lambda[S_col_k]]--;
                            count[lambda[S_col_k] + 1]++;

                            ptr[lambda[S_col_k] + 1]
                                = ptr[lambda[S_col_k]] + count[lambda[S_col_k]];

                            lambda[S_col_k]++;
                        }
                    }
                }
            }

            int S_row_start = csr_row_ptr_S[current_node];
            int S_row_end   = csr_row_ptr_S[current_node + 1];

            for(int j = S_row_start; j < S_row_end; j++)
            {
                int S_col_j = csr_col_ind_S[j];

                if(cfpoints[S_col_j] == U_POINT && lambda[S_col_j] > 0)
                {

                    // // Move j to the beginning of its current interval
                    // I lambda_j = lambda[j];
                    // I old_pos  = node_to_index[j];
                    // I new_pos  = interval_ptr[lambda_j];

                    // node_to_index[index_to_node[old_pos]] = new_pos;
                    // node_to_index[index_to_node[new_pos]] = old_pos;
                    // std::swap(index_to_node[old_pos],index_to_node[new_pos]);

                    // // Update intervals
                    // interval_count[lambda_j]   -= 1;
                    // interval_count[lambda_j-1] += 1;
                    // interval_ptr[lambda_j]     += 1;
                    // interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

                    // // Decrement lambda_j
                    // lambda[j]--;

                    int old_pos = n2i[S_col_j];
                    int new_pos = ptr[lambda[S_col_j]];

                    n2i[i2n[old_pos]] = new_pos;
                    n2i[i2n[new_pos]] = old_pos;

                    std::swap(i2n[old_pos], i2n[new_pos]);

                    count[lambda[S_col_j]]--;
                    count[lambda[S_col_j] - 1]++;

                    ptr[lambda[S_col_j]]++;

                    lambda[S_col_j]--;
                }
            }
        }
    }

    // All remaining unassigned nodes are marked as F-points
    for(int i = 0; i < S.get_m(); i++)
    {
        if(cfpoints[i] == U_POINT)
        {
            cfpoints[i] = F_POINT;
        }
    }
}

void linalg::host_compute_cfpoint_second_pass(const csr_matrix& S, vector<uint32_t>& cfpoints)
{
    ROUTINE_TRACE("linalg::host_compute_cfpoint_second_pass");

    const int* csr_row_ptr_S = S.get_row_ptr();
    const int* csr_col_ind_S = S.get_col_ind();

    for(int i = 0; i < S.get_m(); i++)
    {
        if(cfpoints[i] == F_POINT)
        {
            int S_row_start_i = csr_row_ptr_S[i];
            int S_row_end_i   = csr_row_ptr_S[i + 1];

            int candidate_cpoint = -1;

            for(int j = S_row_start_i; j < S_row_end_i; j++)
            {
                int S_col_j = csr_col_ind_S[j];

                if(cfpoints[S_col_j] == F_POINT)
                {
                    int S_row_start_col_j = csr_row_ptr_S[S_col_j];
                    int S_row_end_col_j   = csr_row_ptr_S[S_col_j + 1];

                    int index1 = S_row_start_i;
                    int index2 = S_row_start_col_j;

                    bool common_cpoint_found = false;
                    while(index1 < S_row_end_i && index2 < S_row_end_col_j)
                    {
                        int col1 = csr_col_ind_S[index1];
                        int col2 = csr_col_ind_S[index2];

                        if(col1 < col2)
                        {
                            index1++;
                        }
                        else if(col2 < col1)
                        {
                            index2++;
                        }
                        else
                        {
                            assert(col1 == col2);

                            index1++;
                            index2++;
                            if(cfpoints[col1] == C_POINT)
                            {
                                common_cpoint_found = true;
                                break;
                            }
                        }
                    }

                    if(!common_cpoint_found)
                    {
                        if(candidate_cpoint < 0)
                        {
                            // If no candidate cpoint has been marked yet, then mark S_col_j as candidate cpoint
                            candidate_cpoint  = S_col_j;
                            cfpoints[S_col_j] = C_POINT;
                        }
                        else
                        {
                            // If a candidate cpoint was previosuly marked, move it back to being an fpoint
                            // and mark S_col_j as the new candidate cpoint
                            cfpoints[candidate_cpoint] = F_POINT;
                            candidate_cpoint           = S_col_j;
                            cfpoints[S_col_j]          = C_POINT;
                        }
                    }
                }
            }
        }
    }
}
