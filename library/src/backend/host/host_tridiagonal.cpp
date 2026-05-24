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
#include "linalg_enums.h"

#include "host_tridiagonal.h"

static constexpr int MAX_RECURSION_LEVELS = 3;

struct linalg::tridiagonal_descr
{
    pivoting_strategy pivoting_strategy;

    // Buffers for non-pivoting approach (one per recursion level)
    double* lower_modified[MAX_RECURSION_LEVELS];
    double* main_modified[MAX_RECURSION_LEVELS];
    double* upper_modified[MAX_RECURSION_LEVELS];
    double* B_modified[MAX_RECURSION_LEVELS];

    double* spike_lower[MAX_RECURSION_LEVELS];
    double* spike_main[MAX_RECURSION_LEVELS];
    double* spike_upper[MAX_RECURSION_LEVELS];
    double* spike_B[MAX_RECURSION_LEVELS];
    double* spike_X[MAX_RECURSION_LEVELS];

    // Buffers for partial pivoting approach (to be implemented)
    double* lower_pad;
    double* main_pad;
    double* upper_pad;
    double* B_pad;

    double* w_pad;
    double* v_pad;

    double* mt;

    double* S_lower;
    double* S_main;
    double* S_upper;
    double* S_B;
};

namespace linalg
{
    template <typename T>
    static void host_thomas_algorithm_impl(int      m,
                                           int      n,
                                           const T* lower_diag,
                                           const T* main_diag,
                                           const T* upper_diag,
                                           const T* b,
                                           T*       x)
    {
        ROUTINE_TRACE("host_thomas_algorithm_impl");
        // T c_prime[M];
        std::vector<T> c_prime(m);
        c_prime[0] = upper_diag[0] / main_diag[0];
        for(int i = 1; i < m - 1; i++)
        {
            T denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
            c_prime[i] = upper_diag[i] / denom;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int j = 0; j < n; j++)
        {
            //T d_prime[M];
            std::vector<T> d_prime(m);
            d_prime[0] = b[m * j + 0] / main_diag[0];
            for(int i = 1; i < m; i++)
            {
                T num      = b[m * j + i] - lower_diag[i] * d_prime[i - 1];
                T denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
                d_prime[i] = num / denom;
            }
            x[m * j + (m - 1)] = d_prime[m - 1];
            for(int i = m - 2; i >= 0; i--)
                x[m * j + i] = d_prime[i] - c_prime[i] * x[m * j + (i + 1)];
        }
    }

    template <typename T>
    bool bunch_kaufman_criterion(T ak_1, T ak_2, T bk, T bk_1, T ck, T ck_1)
    {
        double kappa = double(0.5) * (sqrt(double(5.0)) - double(1.0));

        double sigma = double(0);
        sigma        = std::max(double(abs(ak_1)), double(abs(ak_2)));
        sigma        = std::max(double(abs(bk_1)), sigma);
        sigma        = std::max(double(abs(ck)), sigma);
        sigma        = std::max(double(abs(ck_1)), sigma);

        return abs(bk) * sigma >= kappa * abs(ak_1 * ck);
    }

    template <typename T>
    static void host_spike_algorithm_impl(int      m,
                                          int      n,
                                          const T* lower_diag,
                                          const T* main_diag,
                                          const T* upper_diag,
                                          const T* b,
                                          T*       x)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m * n; i++)
        {
            x[i] = b[i];
        }

        std::vector<T>   mt(m);
        std::vector<int> pivot_mask(m);

        int k  = 0;
        T   bk = main_diag[k];

        while(k < m)
        {
            T ck   = upper_diag[k];
            T ck_1 = (k < (m - 1)) ? upper_diag[k + 1] : static_cast<T>(0);
            T bk_1 = (k < (m - 1)) ? main_diag[k + 1] : static_cast<T>(0);
            T ak_1 = (k < (m - 1)) ? lower_diag[k + 1] : static_cast<T>(0);
            T ak_2 = (k < (m - 2)) ? lower_diag[k + 2] : static_cast<T>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (m - 1))
            {
                const T inv_bk = static_cast<T>(1) / bk;

                mt[k] = ck * inv_bk;

                pivot_mask[k] = 1; // mark this pivot as 1x1

                // L * B * x = y
                for(int i = 0; i < n; i++)
                {
                    T rhsk = x[k + m * i] * inv_bk;

                    x[k + m * i] = rhsk;

                    if(k < (m - 1))
                    {
                        x[k + 1 + m * i] += -(ak_1 * rhsk);
                    }
                }

                if(k < (m - 1))
                {
                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                mt[k] = -ck * ck_1 * det;

                pivot_mask[k] = 2;

                if(k < (m - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;

                    pivot_mask[k + 1] = 2;
                }

                T bk_2 = static_cast<T>(0);

                // L * B * M^T * x = b

                // y = M^T * x
                // L * B * y = b

                // L * B * x = y
                for(int i = 0; i < n; i++)
                {
                    T rhsk   = x[k + m * i] * det;
                    T rhsk_1 = x[k + 1 + m * i] * det;

                    x[k + m * i]     = (bk_1 * rhsk - ck * rhsk_1);
                    x[k + 1 + m * i] = (-ak_1 * rhsk + bk * rhsk_1);

                    if(k < (m - 2))
                    {
                        x[k + 2 + m * i] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
                    }
                }

                if(k < (m - 2))
                {
                    bk_2 = main_diag[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == m);
        // at this point k = m. Could just set k = m - 1 here
        k--;

        k -= pivot_mask[k];

        // backward solve (M^T * x = x)
        while(k >= 0)
        {
            if(pivot_mask[k] == 1)
            {
                const T tmp = mt[k];

                for(int i = 0; i < n; i++)
                {
                    x[k + m * i] += -tmp * x[k + 1 + m * i];
                }

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[k];
                const T tmp2 = mt[k - 1];

                for(int i = 0; i < n; i++)
                {
                    x[k + m * i] += -tmp1 * x[k + 1 + m * i];
                    x[k - 1 + m * i] += -tmp2 * x[k + 1 + m * i];
                }

                k -= 2;
            }
        }
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

    switch(descr->pivoting_strategy)
    {
    case pivoting_strategy::none:
    {
        host_thomas_algorithm_impl(m,
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
        host_spike_algorithm_impl(m,
                                  n,
                                  lower_diag.get_vec(),
                                  main_diag.get_vec(),
                                  upper_diag.get_vec(),
                                  b.get_vec(),
                                  x.get_vec());
        break;
    }
    }
}
