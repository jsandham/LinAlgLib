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

#ifndef AMG_KERNELS_H
#define AMG_KERNELS_H

#include "common.cuh"

static __device__ unsigned int hash1(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x / 2;
}

template <uint32_t BLOCKSIZE>
__global__ void initialize_pmis_state_kernel(int m,
                                             int n,
                                             int nnz,
                                             const int* __restrict__ csr_row_ptr,
                                             const int* __restrict__ connections,
                                             int* __restrict__ state,
                                             int* __restrict__ hash)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m; row += BLOCKSIZE * gridDim.x)
    {
        int s = -2;

        int row_start = csr_row_ptr[row];
        int row_end   = csr_row_ptr[row + 1];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                s = 0;
                break;
            }
        }

        state[row] = s;
        hash[row]  = hash1(row);
    }
}

struct pmis_node
{
    int state;
    int hash;
    int row;
};

static __device__ pmis_node lexographical_max(const pmis_node* ti, const pmis_node* tj)
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

template <uint32_t BLOCKSIZE>
__global__ void find_maximum_distance_two_node_kernel(int m,
                                                      int n,
                                                      int nnz,
                                                      const int* __restrict__ csr_row_ptr,
                                                      const int* __restrict__ csr_col_ind,
                                                      const int* __restrict__ connections,
                                                      const int* __restrict__ state,
                                                      const int* __restrict__ hash,
                                                      int64_t* __restrict__ aggregates,
                                                      int* __restrict__ max_state,
                                                      bool* __restrict__ complete)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m; row += BLOCKSIZE * gridDim.x)
    {
        pmis_node max_node;
        max_node.state = state[row];
        max_node.hash  = hash[row];
        max_node.row   = row;

        int row_start = csr_row_ptr[row];
        int row_end   = csr_row_ptr[row + 1];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                int col = csr_col_ind[j];

                pmis_node node;
                node.state = state[col];
                node.hash  = hash[col];
                node.row   = col;

                max_node = lexographical_max(&max_node, &node);
            }
        }

        // Find distance 2 maximum neighbour node
        int row_start2 = csr_row_ptr[max_node.row];
        int row_end2   = csr_row_ptr[max_node.row + 1];

        for(int j = row_start2; j < row_end2; j++)
        {
            if(connections[j] == 1)
            {
                int col = csr_col_ind[j];

                pmis_node node;
                node.state = state[col];
                node.hash  = hash[col];
                node.row   = col;

                max_node = lexographical_max(&max_node, &node);
            }
        }

        if(state[row] == 0)
        {
            // If max node is current node, then make current node an aggregate root.
            if(max_node.row == row)
            {
                max_state[row]  = 1;
                aggregates[row] = 1;
            }
            // If max node is not current node, but max node has a state of 1, then
            // the max node must be an already existing aggregate root and therefore
            // the current node is too close to an existing aggregate root for it to
            // also be an aggregate root. We mark it with state -1 to indicate it
            // cannot be an aggregate root.
            else if(max_node.state == 1)
            {
                max_state[row]  = -1;
                aggregates[row] = 0;
            }
            // If max node is not current node, and also does not have a state of 1,
            // then we must call this function again so we mark the work as not
            // complete.
            else
            {
                *complete = false;
            }
        }
    }
}

template <uint32_t BLOCKSIZE>
__global__ void
    add_unassigned_nodes_to_closest_aggregation_kernel(int m,
                                                       int n,
                                                       int nnz,
                                                       const int* __restrict__ csr_row_ptr,
                                                       const int* __restrict__ csr_col_ind,
                                                       const int* __restrict__ connections,
                                                       const int* __restrict__ state,
                                                       int64_t* __restrict__ aggregates,
                                                       int64_t* __restrict__ aggregate_root_nodes,
                                                       int* __restrict__ max_state)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m; row += BLOCKSIZE * gridDim.x)
    {
        if(state[row] == -1)
        {
            int start = csr_row_ptr[row];
            int end   = csr_row_ptr[row + 1];

            for(int j = start; j < end; j++)
            {
                if(connections[j] == 1)
                {
                    int col = csr_col_ind[j];

                    if(state[col] == 1)
                    {
                        aggregates[row] = aggregates[col];
                        max_state[row]  = 1;
                        break;
                    }
                }
            }
        }
        else if(state[row] == -2)
        {
            aggregates[row] = -2;
        }
    }
}

#endif