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
#include <iostream>

#include "device_amg_aggregation.h"

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_amg_aggregation.h"
#endif

void linalg::device_initialize_pmis_state(const csr_matrix&  A,
                                          const vector<int>& connections,
                                          vector<int>&       state,
                                          vector<int>&       hash)
{
    ROUTINE_TRACE("linalg::device_initialize_pmis_state");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_initialize_pmis_state(A.get_m(),
                                             A.get_n(),
                                             A.get_nnz(),
                                             A.get_row_ptr(),
                                             connections.get_vec(),
                                             state.get_vec(),
                                             hash.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // device_initialize_pmis_state_impl(A.get_m(),
    //                                   A.get_n(),
    //                                   A.get_nnz(),
    //                                   A.get_row_ptr(),
    //                                   connections.get_vec(),
    //                                   state.get_vec(),
    //                                   hash.get_vec());
}

void linalg::device_find_maximum_distance_two_node(const csr_matrix&  A,
                                                   const vector<int>& connections,
                                                   const vector<int>& state,
                                                   const vector<int>& hash,
                                                   vector<int64_t>&   aggregates,
                                                   vector<int>&       max_state,
                                                   bool&              complete)
{
    ROUTINE_TRACE("linalg::device_find_maximum_distance_two_node");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_find_maximum_distance_two_node(A.get_m(),
                                                      A.get_n(),
                                                      A.get_nnz(),
                                                      A.get_row_ptr(),
                                                      A.get_col_ind(),
                                                      connections.get_vec(),
                                                      state.get_vec(),
                                                      hash.get_vec(),
                                                      aggregates.get_vec(),
                                                      max_state.get_vec(),
                                                      complete));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // device_find_maximum_distance_two_node_impl(A.get_m(),
    //                                            A.get_n(),
    //                                            A.get_nnz(),
    //                                            A.get_row_ptr(),
    //                                            A.get_col_ind(),
    //                                            connections.get_vec(),
    //                                            state.get_vec(),
    //                                            hash.get_vec(),
    //                                            aggregates.get_vec(),
    //                                            max_state.get_vec(),
    //                                            complete);
}

void linalg::device_add_unassigned_nodes_to_closest_aggregation(
    const csr_matrix&  A,
    const vector<int>& connections,
    const vector<int>& state,
    vector<int64_t>&   aggregates,
    vector<int64_t>&   aggregate_root_nodes,
    vector<int>&       max_state)
{
    ROUTINE_TRACE("linalg::device_add_unassigned_nodes_to_closest_aggregation");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_add_unassigned_nodes_to_closest_aggregation(A.get_m(),
                                                                   A.get_n(),
                                                                   A.get_nnz(),
                                                                   A.get_row_ptr(),
                                                                   A.get_col_ind(),
                                                                   connections.get_vec(),
                                                                   state.get_vec(),
                                                                   aggregates.get_vec(),
                                                                   aggregate_root_nodes.get_vec(),
                                                                   max_state.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // device_add_unassigned_nodes_to_closest_aggregation_impl(A.get_m(),
    //                                                         A.get_n(),
    //                                                         A.get_nnz(),
    //                                                         A.get_row_ptr(),
    //                                                         A.get_col_ind(),
    //                                                         connections.get_vec(),
    //                                                         state.get_vec(),
    //                                                         aggregates.get_vec(),
    //                                                         aggregate_root_nodes.get_vec(),
    //                                                         max_state.get_vec());
}

void linalg::device_compute_cfpoint_first_pass(const csr_matrix& S,
                                               const csr_matrix& ST,
                                               vector<uint32_t>& cfpoints)
{
}

void linalg::device_compute_cfpoint_second_pass(const csr_matrix& S, vector<uint32_t>& cfpoints) {}