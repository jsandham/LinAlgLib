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

#ifndef DEVICE_AMG_AGGREGATION_H
#define DEVICE_AMG_AGGREGATION_H

#include <cstdint>
#include <string>

#include "csr_matrix.h"
#include "linalg_export.h"
#include "vector.h"

namespace linalg
{
    //
    void device_initialize_pmis_state(const csr_matrix&  A,
                                      const vector<int>& connections,
                                      vector<int>&       state,
                                      vector<int>&       hash);

    void device_find_maximum_distance_two_node(const csr_matrix&  A,
                                               const vector<int>& connections,
                                               const vector<int>& state,
                                               const vector<int>& hash,
                                               vector<int64_t>&   aggregates,
                                               vector<int>&       max_state,
                                               bool&              complete);

    void device_add_unassigned_nodes_to_closest_aggregation(const csr_matrix&  A,
                                                            const vector<int>& connections,
                                                            const vector<int>& state,
                                                            vector<int64_t>&   aggregates,
                                                            vector<int64_t>&   aggregate_root_nodes,
                                                            vector<int>&       max_state);

    void device_compute_cfpoint_first_pass(const csr_matrix& S,
                                           const csr_matrix& ST,
                                           vector<uint32_t>& cfpoints);

    void device_compute_cfpoint_second_pass(const csr_matrix& S, vector<uint32_t>& cfpoints);
}

#endif