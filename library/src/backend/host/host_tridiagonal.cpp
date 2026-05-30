//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#include <cassert>
#include <iostream>

#include "../../trace.h"

#include "../../descriptors/tridiagonal_descr_internal.h"

#include "host_tridiagonal.h"
#include "spike_algorithm.h"
#include "thomas_algorithm.h"

namespace linalg
{
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

void linalg::host_tridiagonal_analysis(int                   m,
                                       int                   n,
                                       const vector<double>& lower_diag,
                                       const vector<double>& main_diag,
                                       const vector<double>& upper_diag,
                                       tridiagonal_descr*    descr)
{
    ROUTINE_TRACE("linalg::host_tridiagonal_allocate_buffers");
    assert(m > 0);
    assert(n > 0);
    assert(main_diag.get_size() == m);
    assert(lower_diag.get_size() == m);
    assert(upper_diag.get_size() == m);

    descr->host_analysis_valid    = true;
    descr->host_analysis_m        = m;
    descr->host_analysis_n        = n;
    descr->host_analysis_strategy = descr->strategy;

    switch(descr->strategy)
    {
    case pivoting_strategy::none:
    {
        // No additional buffers needed for Thomas algorithm.
        break;
    }
    case pivoting_strategy::partial:
    {
        const int m_pad = static_cast<int>(next_power_of_two(static_cast<uint64_t>(m)));

        std::cout << "analysis m: " << m << ", m_pad: " << m_pad << std::endl;

        descr->host_data.lower_pad.resize(m_pad, 0.0);
        descr->host_data.main_pad.resize(m_pad, 1.0);
        descr->host_data.upper_pad.resize(m_pad, 0.0);
        descr->host_data.B_pad.resize(m_pad * n, 0.0);

        descr->host_data.w_pad.resize(m_pad, 0.0);
        descr->host_data.v_pad.resize(m_pad, 0.0);
        descr->host_data.mt.resize(m_pad, 0.0);

        constexpr int BLOCKDIM = 8;
        const int     S_size   = 2 * m_pad / BLOCKDIM;

        descr->host_data.S_lower.resize(S_size, 0.0);
        descr->host_data.S_main.resize(S_size, 1.0);
        descr->host_data.S_upper.resize(S_size, 0.0);
        descr->host_data.S_B.resize(S_size * n, 0.0);
        break;
    }
    }
}

void linalg::host_tridiagonal_solver(int                      m,
                                     int                      n,
                                     const vector<double>&    lower_diag,
                                     const vector<double>&    main_diag,
                                     const vector<double>&    upper_diag,
                                     const vector<double>&    b,
                                     vector<double>&          x,
                                     const tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::host_tridiagonal_solver");
    assert(main_diag.get_size() == m);
    assert(lower_diag.get_size() == m);
    assert(upper_diag.get_size() == m);
    assert(b.get_size() == m * n);
    assert(x.get_size() == m * n);

    switch(descr->strategy)
    {
    case pivoting_strategy::none:
    {
        thomas_algorithm_template(m,
                                  n,
                                  lower_diag.get_vec(),
                                  main_diag.get_vec(),
                                  upper_diag.get_vec(),
                                  b.get_vec(),
                                  x.get_vec());
        break;
    }
    case pivoting_strategy::partial:
    {
        spike_algorithm_template(m,
                                 n,
                                 lower_diag.get_vec(),
                                 main_diag.get_vec(),
                                 upper_diag.get_vec(),
                                 b.get_vec(),
                                 x.get_vec(),
                                 descr);
        break;
    }
    }
}
