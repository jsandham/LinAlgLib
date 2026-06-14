//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#include "../../../include/direct_solvers/tridiagonal/tridiagonal.h"

#include "../../trace.h"
#include "../../utility.h"

#include "../../backend/device/device_math.h"
#include "../../backend/host/host_math.h"

namespace linalg
{
    static void non_pivoting_algorithm(int                   m,
                                       int                   n,
                                       const vector<double>& lower_diag,
                                       const vector<double>& main_diag,
                                       const vector<double>& upper_diag,
                                       const vector<double>& rhs,
                                       vector<double>&       solution,
                                       non_pivoting_data&    non_pivot_data)
    {
        ROUTINE_TRACE("linalg::non_pivoting_algorithm");

        backend_dispatch("non_pivoting_algorithm",
                         host_non_pivoting_algorithm,
                         device_non_pivoting_algorithm,
                         m,
                         n,
                         lower_diag,
                         main_diag,
                         upper_diag,
                         rhs,
                         solution,
                         non_pivot_data);
    }

    static void partial_pivoting_algorithm(int                   m,
                                           int                   n,
                                           const vector<double>& lower_diag,
                                           const vector<double>& main_diag,
                                           const vector<double>& upper_diag,
                                           const vector<double>& rhs,
                                           vector<double>&       solution,
                                           pivoting_data&        pivot_data)
    {
        ROUTINE_TRACE("linalg::partial_pivoting_algorithm");

        backend_dispatch("partial_pivoting_algorithm",
                         host_partial_pivoting_algorithm,
                         device_partial_pivoting_algorithm,
                         m,
                         n,
                         lower_diag,
                         main_diag,
                         upper_diag,
                         rhs,
                         solution,
                         pivot_data);
    }

    static uint64_t next_power_of_two(uint64_t m)
    {
        if(m == 0)
        {
            return 1;
        }

        m--;

        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        m |= m >> 16;
        m |= m >> 32;

        return m + 1;
    }
}

linalg::tridiagonal_solver::tridiagonal_solver(int m, int n, pivoting_strategy strategy)
    : m(m)
    , n(n)
    , strategy(strategy)
    , on_host(true)
{
    switch(strategy)
    {
    case pivoting_strategy::none:
    {
        constexpr int BLOCKSIZE = 256;
        int           current_m = m;
        for(int level = 0; level < non_pivoting_data::tridiagonal_max_recursion_levels; level++)
        {
            if(current_m <= 1024)
                break;

            int nblocks    = ((current_m - 1) / BLOCKSIZE + 1);
            int num_spikes = 2 * nblocks;

            non_pivot_data.lower_modified[level].resize(current_m);
            non_pivot_data.main_modified[level].resize(current_m);
            non_pivot_data.upper_modified[level].resize(current_m);
            non_pivot_data.B_modified[level].resize(current_m * n);
            non_pivot_data.spike_lower[level].resize(num_spikes);
            non_pivot_data.spike_main[level].resize(num_spikes);
            non_pivot_data.spike_upper[level].resize(num_spikes);
            non_pivot_data.spike_B[level].resize(num_spikes * n);
            non_pivot_data.spike_X[level].resize(num_spikes * n);

            current_m = num_spikes;
        }
        break;
    }
    case pivoting_strategy::partial:
    {
        constexpr int BLOCKDIM = pivoting_data::block_dim;

        int current_m = m;
        for(int level = 0; level < non_pivoting_data::tridiagonal_max_recursion_levels; level++)
        {
            std::cout << "level: " << level << " current_m: " << current_m << std::endl;
            //if(current_m <= 1024)
            //    break;

            int m_pad = static_cast<int>(next_power_of_two(static_cast<uint64_t>(current_m)));
            m_pad     = std::max(m_pad, BLOCKDIM);

            // For partial pivoting, we would initialize the padded buffers here.
            pivot_data.lower_pad[level].resize(m_pad);
            pivot_data.main_pad[level].resize(m_pad);
            pivot_data.upper_pad[level].resize(m_pad);
            pivot_data.B_pad[level].resize(m_pad * n);
            pivot_data.w[level].resize(m_pad);
            pivot_data.v[level].resize(m_pad);
            pivot_data.mt[level].resize(m_pad);

            const int S_size = 2 * m_pad / BLOCKDIM;

            pivot_data.S_lower[level].resize(S_size);
            pivot_data.S_main[level].resize(S_size);
            pivot_data.S_upper[level].resize(S_size);
            pivot_data.S_B[level].resize(S_size * n);

            current_m = S_size;
        }
        break;
    }
    }
}

linalg::tridiagonal_solver::~tridiagonal_solver()
{
    // No dynamic memory to free in this implementation, but if we had device buffers, we would free them here.
}

void linalg::tridiagonal_solver::move_to_device()
{
    if(on_host)
    {
        for(int i = 0; i < non_pivoting_data::tridiagonal_max_recursion_levels; i++)
        {
            non_pivot_data.lower_modified[i].move_to_device();
            non_pivot_data.main_modified[i].move_to_device();
            non_pivot_data.upper_modified[i].move_to_device();
            non_pivot_data.B_modified[i].move_to_device();
            non_pivot_data.spike_lower[i].move_to_device();
            non_pivot_data.spike_main[i].move_to_device();
            non_pivot_data.spike_upper[i].move_to_device();
            non_pivot_data.spike_B[i].move_to_device();
            non_pivot_data.spike_X[i].move_to_device();
        }

        for(int i = 0; i < pivoting_data::tridiagonal_max_recursion_levels; i++)
        {
            pivot_data.lower_pad[i].move_to_device();
            pivot_data.main_pad[i].move_to_device();
            pivot_data.upper_pad[i].move_to_device();
            pivot_data.B_pad[i].move_to_device();
            pivot_data.w[i].move_to_device();
            pivot_data.v[i].move_to_device();
            pivot_data.mt[i].move_to_device();
            pivot_data.S_lower[i].move_to_device();
            pivot_data.S_main[i].move_to_device();
            pivot_data.S_upper[i].move_to_device();
            pivot_data.S_B[i].move_to_device();
        }

        on_host = false;
    }
}

void linalg::tridiagonal_solver::move_to_host()
{
    if(!on_host)
    {
        for(int i = 0; i < non_pivoting_data::tridiagonal_max_recursion_levels; i++)
        {
            non_pivot_data.lower_modified[i].move_to_host();
            non_pivot_data.main_modified[i].move_to_host();
            non_pivot_data.upper_modified[i].move_to_host();
            non_pivot_data.B_modified[i].move_to_host();
            non_pivot_data.spike_lower[i].move_to_host();
            non_pivot_data.spike_main[i].move_to_host();
            non_pivot_data.spike_upper[i].move_to_host();
            non_pivot_data.spike_B[i].move_to_host();
            non_pivot_data.spike_X[i].move_to_host();
        }

        for(int i = 0; i < pivoting_data::tridiagonal_max_recursion_levels; i++)
        {
            pivot_data.lower_pad[i].move_to_host();
            pivot_data.main_pad[i].move_to_host();
            pivot_data.upper_pad[i].move_to_host();
            pivot_data.B_pad[i].move_to_host();
            pivot_data.w[i].move_to_host();
            pivot_data.v[i].move_to_host();
            pivot_data.mt[i].move_to_host();
            pivot_data.S_lower[i].move_to_host();
            pivot_data.S_main[i].move_to_host();
            pivot_data.S_upper[i].move_to_host();
            pivot_data.S_B[i].move_to_host();
        }

        on_host = true;
    }
}

void linalg::tridiagonal_solver::solve(const vector<double>& lower_diag,
                                       const vector<double>& main_diag,
                                       const vector<double>& upper_diag,
                                       const vector<double>& rhs,
                                       vector<double>&       solution)
{
    switch(strategy)
    {
    case pivoting_strategy::none:
        non_pivoting_algorithm(
            m, n, lower_diag, main_diag, upper_diag, rhs, solution, non_pivot_data);
        break;
    case pivoting_strategy::partial:
        partial_pivoting_algorithm(
            m, n, lower_diag, main_diag, upper_diag, rhs, solution, pivot_data);
        break;
    }
}
