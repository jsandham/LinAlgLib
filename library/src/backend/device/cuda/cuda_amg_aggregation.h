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
#ifndef CUDA_AMG_STRENGTH_H
#define CUDA_AMG_STRENGTH_H

namespace linalg
{
    void cuda_initialize_pmis_state(int        m,
                                    int        n,
                                    int        nnz,
                                    const int* csr_row_ptr,
                                    const int* connections,
                                    int*       state,
                                    int*       hash);

    void cuda_find_maximum_distance_two_node(int        m,
                                             int        n,
                                             int        nnz,
                                             const int* csr_row_ptr,
                                             const int* csr_col_ind,
                                             const int* connections,
                                             const int* state,
                                             const int* hash,
                                             int64_t*   aggregates,
                                             int*       max_state,
                                             bool&      complete);

    void cuda_add_unassigned_nodes_to_closest_aggregation(int        m,
                                                          int        n,
                                                          int        nnz,
                                                          const int* csr_row_ptr,
                                                          const int* csr_col_ind,
                                                          const int* connections,
                                                          const int* state,
                                                          int64_t*   aggregates,
                                                          int64_t*   aggregate_root_nodes,
                                                          int*       max_state);
}

#endif