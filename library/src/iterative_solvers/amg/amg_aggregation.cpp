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

#include "iterative_solvers/amg/amg_aggregation.h"
#include "iterative_solvers/amg/amg.h"
#include "linalg.h"
#include <assert.h>
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#include "../../backend/device/device_amg_aggregation.h"
#include "../../backend/host/host_amg_aggregation.h"

//********************************************************************************
//
// amg: Smoothed Aggregation Algebraic Multigrid
//
//********************************************************************************

namespace linalg
{
    static void initialize_pmis_state(const csr_matrix&  A,
                                      const vector<int>& connections,
                                      vector<int>&       state,
                                      vector<int>&       hash)
    {
        ROUTINE_TRACE("initialize_pmis_state");

        backend_dispatch("initialize_pmis_state",
                         host_initialize_pmis_state,
                         device_initialize_pmis_state,
                         A,
                         connections,
                         state,
                         hash);
    }

    static void find_maximum_distance_two_node(const csr_matrix&  A,
                                               const vector<int>& connections,
                                               const vector<int>& state,
                                               const vector<int>& hash,
                                               vector<int64_t>&   aggregates,
                                               vector<int>&       max_state,
                                               bool&              complete)
    {
        ROUTINE_TRACE("find_maximum_distance_two_node");

        backend_dispatch("find_maximum_distance_two_node",
                         host_find_maximum_distance_two_node,
                         device_find_maximum_distance_two_node,
                         A,
                         connections,
                         state,
                         hash,
                         aggregates,
                         max_state,
                         complete);
    }

    static void add_unassigned_nodes_to_closest_aggregation(const csr_matrix&  A,
                                                            const vector<int>& connections,
                                                            const vector<int>& state,
                                                            vector<int64_t>&   aggregates,
                                                            vector<int64_t>&   aggregate_root_nodes,
                                                            vector<int>&       max_state)
    {
        ROUTINE_TRACE("add_unassigned_nodes_to_closest_aggregation");

        backend_dispatch("add_unassigned_nodes_to_closest_aggregation",
                         host_add_unassigned_nodes_to_closest_aggregation,
                         device_add_unassigned_nodes_to_closest_aggregation,
                         A,
                         connections,
                         state,
                         aggregates,
                         aggregate_root_nodes,
                         max_state);
    }
}

bool linalg::compute_aggregates_using_pmis(const csr_matrix&  A,
                                           const vector<int>& connections,
                                           vector<int64_t>&   aggregates,
                                           vector<int64_t>&   aggregate_root_nodes)
{
    ROUTINE_TRACE("linalg::compute_aggregates_using_pmis");

    backend bend = determine_backend(A, connections, aggregates, aggregate_root_nodes);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::compute_aggregates_using_pmis "
                     "must all be "
                     "on host or "
                     "all be on device"
                  << std::endl;
        return false;
    }

    //connections.print_vector("connections");

    vector<int> hash(A.get_m());
    vector<int> state(A.get_m());
    vector<int> max_state(A.get_m());

    if(bend == backend::device)
    {
        hash.move_to_device();
        state.move_to_device();
        max_state.move_to_device();
    }

    // Initialize parallel maximal independent set state
    initialize_pmis_state(A, connections, max_state, hash);

    //max_state.print_vector("matrix_state");
    //hash.print_vector("hash");

    int iter = 0;
    while(iter < 20)
    {
        state.copy_from(max_state);

        // Find maximum distance 2 node
        bool complete = true;
        find_maximum_distance_two_node(
            A, connections, state, hash, aggregates, max_state, complete);

        if(complete)
        {
            break;
        }

        if(iter > 20)
        {
            std::cout << "Hit maximum iterations when determinig aggregates" << std::endl;
            break;
        }

        iter++;
    }

    std::cout << "A" << std::endl;

    aggregate_root_nodes.resize(A.get_m(), -1);
    aggregate_root_nodes.fill(-1);

    std::cout << "B" << std::endl;

    //aggregate_root_nodes.move_to_host();
    //aggregates.move_to_host();

    for(size_t i = 0; i < aggregates.get_size(); i++)
    {
        aggregate_root_nodes[i] = (aggregates[i] == 1) ? 1 : -1;
    }
    std::cout << "C" << std::endl;

    //aggregate_root_nodes.move_to_device();
    //aggregates.move_to_device();

    //aggregates.print_vector("aggregates before exclusive sum");

    // Exclusive sum
    exclusive_scan(aggregates);

    if(aggregates.get_size() < 300)
    {
        aggregates.print_vector("aggregates after scan");
    }

    //max_state.print_vector("max_state");
    //aggregates.print_vector("aggregates after exclusive sum");
    //aggregate_root_nodes.print_vector("aggregate_root_nodes");

    std::cout << "D" << std::endl;
    // Add any unassigned nodes to an existing aggregation
    for(int k = 0; k < 2; k++)
    {
        state.copy_from(max_state);

        add_unassigned_nodes_to_closest_aggregation(
            A, connections, state, aggregates, aggregate_root_nodes, max_state);
    }

    std::cout << "E" << std::endl;

    if(aggregates.get_size() < 300)
    {
        aggregates.print_vector("aggregates final");
        aggregate_root_nodes.print_vector("aggregate_root_nodes final");
    }

    return true;
}

//-------------------------------------------------------------------------------
// function for finding c-points and f-points (first pass)
//-------------------------------------------------------------------------------
#define F_POINT 0 // F-point
#define C_POINT 1 // C-point
#define U_POINT 2 // Unassigned point
#define FF_POINT 3 // Future F-point

void linalg::compute_cfpoint_first_pass(const csr_matrix& S,
                                        const csr_matrix& ST,
                                        vector<uint32_t>& cfpoints)
{
    ROUTINE_TRACE("linalg::compute_cfpoint_first_pass");

    backend_dispatch("compute_cfpoint_first_pass",
                     host_compute_cfpoint_first_pass,
                     device_compute_cfpoint_first_pass,
                     S,
                     ST,
                     cfpoints);
}

//-------------------------------------------------------------------------------
// function for finding c-points and f-points (second pass)
//-------------------------------------------------------------------------------
void linalg::compute_cfpoint_second_pass(const csr_matrix& S, vector<uint32_t>& cfpoints)
{
    ROUTINE_TRACE("linalg::compute_cfpoint_second_pass");

    backend_dispatch("compute_cfpoint_second_pass",
                     host_compute_cfpoint_second_pass,
                     device_compute_cfpoint_second_pass,
                     S,
                     cfpoints);
}