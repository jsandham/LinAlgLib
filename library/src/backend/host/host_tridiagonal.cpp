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

#include "../../descriptors/tridiagonal_descr_internal.h"

#include "host_tridiagonal.h"

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

    template <int WORDS>
    struct PivotMask
    {
        unsigned int bits[WORDS];

        // Sets bit k to 0 to record a 1x1 pivot at row k.
        void set_pivoting_to_1x1(int k)
        {
            bits[k >> 5] &= ~(1u << (k & 31));
        }

        // Sets bit k to 1 to record 2x2 pivoting at row k.
        void set_pivoting_to2x2(int k)
        {
            bits[k >> 5] |= (1u << (k & 31));
        }

        // Returns 1 if row k used 1x1 pivoting, 2 if row k is part of a 2x2 pivot.
        int get_pivoting(int k) const
        {
            return ((bits[k >> 5] >> (k & 31)) & 1u) ? 2 : 1;
        }
    };

    template <typename T, uint32_t BLOCKDIM>
    static void LBMT_solve(
        int m_pad, int n, const T* lower, const T* main, const T* upper, T* w, T* v, T* mt, T* rhs)
    {
        const int nblocks = m_pad / BLOCKDIM;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < nblocks; i++)
        {
            T bk = main[i];

            constexpr int               PIVOT_MASK_WORDS = (BLOCKDIM + 31) / 32;
            PivotMask<PIVOT_MASK_WORDS> pivot_mask;

            w[i]                            = lower[i];
            v[i + (BLOCKDIM - 1) * nblocks] = upper[i + (BLOCKDIM - 1) * nblocks];

            int k = 0;
            while(k < BLOCKDIM)
            {
                T ck   = upper[nblocks * k + i];
                T ck_1 = (k < (BLOCKDIM - 1)) ? upper[nblocks * (k + 1) + i] : static_cast<T>(0);
                T bk_1 = (k < (BLOCKDIM - 1)) ? main[nblocks * (k + 1) + i] : static_cast<T>(0);
                T ak_1 = (k < (BLOCKDIM - 1)) ? lower[nblocks * (k + 1) + i] : static_cast<T>(0);
                T ak_2 = (k < (BLOCKDIM - 2)) ? lower[nblocks * (k + 2) + i] : static_cast<T>(0);

                // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
                // pivoting criteria
                const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

                // 1x1 pivoting
                if(use_1x1_pivot || k == (BLOCKDIM - 1))
                {
                    const T inv_bk = static_cast<T>(1) / bk;

                    T wk = w[nblocks * k + i];
                    T vk = v[nblocks * k + i];

                    w[nblocks * k + i]  = wk * inv_bk;
                    v[nblocks * k + i]  = vk * inv_bk;
                    mt[nblocks * k + i] = ck * inv_bk;

                    pivot_mask.set_pivoting_to_1x1(k);

                    if(k < (BLOCKDIM - 1))
                    {
                        w[nblocks * (k + 1) + i] += -ak_1 * wk * inv_bk;
                    }

                    // L * B * x = y
                    for(int j = 0; j < n; j++)
                    {
                        T rhsk = rhs[nblocks * k + i + m_pad * j] * inv_bk;

                        rhs[nblocks * k + i + m_pad * j] = rhsk;

                        if(k < (BLOCKDIM - 1))
                        {
                            rhs[nblocks * (k + 1) + i + m_pad * j] += -(ak_1 * rhsk);
                        }
                    }

                    if(k < (BLOCKDIM - 1))
                    {
                        bk_1 = bk_1 - ak_1 * ck * inv_bk;
                    }

                    bk = bk_1;

                    k += 1;
                }
                else
                {
                    const T det = static_cast<T>(1) / (bk * bk_1 - ak_1 * ck);

                    T wk   = w[nblocks * k + i];
                    T wk_1 = w[nblocks * (k + 1) + i];
                    T vk   = v[nblocks * k + i];
                    T vk_1 = v[nblocks * (k + 1) + i];

                    w[nblocks * k + i]  = (bk_1 * wk - ck * wk_1) * det;
                    v[nblocks * k + i]  = (bk_1 * vk - ck * vk_1) * det;
                    mt[nblocks * k + i] = -ck * ck_1 * det;

                    pivot_mask.set_pivoting_to2x2(k);

                    if(k < (BLOCKDIM - 1))
                    {
                        w[nblocks * (k + 1) + i]  = (-ak_1 * wk + bk * wk_1) * det;
                        v[nblocks * (k + 1) + i]  = (-ak_1 * vk + bk * vk_1) * det;
                        mt[nblocks * (k + 1) + i] = bk * ck_1 * det;

                        pivot_mask.set_pivoting_to2x2(k + 1);
                    }

                    T bk_2 = static_cast<T>(0);

                    if(k < (BLOCKDIM - 2))
                    {
                        w[nblocks * (k + 2) + i] += -(-ak_1 * ak_2 * wk + ak_2 * bk * wk_1) * det;
                    }

                    // |bk   ck  ||xk  |   |rhsk   |
                    // |ak_1 bk_1||xk_1| = |rhsk _1|
                    //
                    //inv = 1 / (bk * bk_1 - ak_1 * ck) |bk_1 -ck  |
                    //                                  |-ak_1  bk |

                    // L * B * x = y
                    for(int j = 0; j < n; j++)
                    {
                        T rhsk   = rhs[nblocks * k + i + m_pad * j] * det;
                        T rhsk_1 = rhs[nblocks * (k + 1) + i + m_pad * j] * det;

                        rhs[nblocks * k + i + m_pad * j]       = (bk_1 * rhsk - ck * rhsk_1);
                        rhs[nblocks * (k + 1) + i + m_pad * j] = (-ak_1 * rhsk + bk * rhsk_1);

                        if(k < (BLOCKDIM - 2))
                        {
                            rhs[nblocks * (k + 2) + i + m_pad * j]
                                += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
                        }
                    }

                    if(k < (BLOCKDIM - 2))
                    {
                        bk_2 = main[nblocks * (k + 2) + i];
                        bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                    }

                    bk = bk_2;
                    k += 2;
                }
            }

            assert(k == BLOCKDIM);
            // at this point k = BLOCKDIM. Could just set k = BLOCKDIM - 1 here
            k--;

            k -= pivot_mask.get_pivoting(k);

            // backward solve (M^T * w = w, M^T * v = v, and M^T * rhs = rhs)
            while(k >= 0)
            {
                if(pivot_mask.get_pivoting(k) == 1)
                {
                    const T tmp = mt[nblocks * k + i];

                    w[nblocks * k + i] += -tmp * w[nblocks * (k + 1) + i];
                    v[nblocks * k + i] += -tmp * v[nblocks * (k + 1) + i];

                    for(int j = 0; j < n; j++)
                    {
                        rhs[nblocks * k + i + m_pad * j]
                            += -tmp * rhs[nblocks * (k + 1) + i + m_pad * j];
                    }

                    k -= 1;
                }
                else
                {
                    const T tmp1 = mt[nblocks * k + i];
                    const T tmp2 = mt[nblocks * (k - 1) + i];

                    w[nblocks * k + i] += -tmp1 * w[nblocks * (k + 1) + i];
                    w[nblocks * (k - 1) + i] += -tmp2 * w[nblocks * (k + 1) + i];
                    v[nblocks * k + i] += -tmp1 * v[nblocks * (k + 1) + i];
                    v[nblocks * (k - 1) + i] += -tmp2 * v[nblocks * (k + 1) + i];

                    for(int j = 0; j < n; j++)
                    {
                        rhs[nblocks * k + i + m_pad * j]
                            += -tmp1 * rhs[nblocks * (k + 1) + i + m_pad * j];
                        rhs[nblocks * (k - 1) + i + m_pad * j]
                            += -tmp2 * rhs[nblocks * (k + 1) + i + m_pad * j];
                    }

                    k -= 2;
                }
            }
        }
    }

    static uint64_t next_power_of_two(uint64_t m)
    {
        // If m is already a power of 2 or 0, return m (or 1 if you prefer 2^0)
        if(m == 0)
            return 1;

        // Decrement m so that if it is already a power of 2,
        // the operations below don't jump it to the next one.
        m--;

        // Fill all bits to the right of the most significant bit with 1s
        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        m |= m >> 16;
        m |= m >> 32; // Include this if using 64-bit integers

        // Adding 1 results in a single bit set at the next power of 2
        return m + 1;
    }

    template <typename T, uint32_t BLOCKDIM>
    static void data_marshalling(int      m,
                                 int      m_pad,
                                 int      n,
                                 const T* lower,
                                 const T* main,
                                 const T* upper,
                                 const T* B,
                                 T*       lower_pad,
                                 T*       main_pad,
                                 T*       upper_pad,
                                 T*       B_pad)
    {
        // Perform data marshalling with padding
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_pad; i++)
        {
            const int gwid = i / (m_pad / BLOCKDIM);
            const int glid = i % (m_pad / BLOCKDIM);

            if(BLOCKDIM * glid + gwid < m)
            {
                lower_pad[i] = lower[BLOCKDIM * glid + gwid];
                main_pad[i]  = main[BLOCKDIM * glid + gwid];
                upper_pad[i] = upper[BLOCKDIM * glid + gwid];

                for(int j = 0; j < n; j++)
                {
                    B_pad[i + m_pad * j] = B[BLOCKDIM * glid + gwid + m * j];
                }
            }
        }
    }

    template <typename T, uint32_t BLOCKDIM>
    static void data_marshalling2(int m, int m_pad, int n, const T* B_pad, T* B)
    {
        // Perform data marshalling with padding
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_pad; i++)
        {
            const int gwid = i / (m_pad / BLOCKDIM);
            const int glid = i % (m_pad / BLOCKDIM);

            if(BLOCKDIM * glid + gwid < m)
            {
                for(int j = 0; j < n; j++)
                {
                    B[BLOCKDIM * glid + gwid + m * j] = B_pad[i + m_pad * j];
                }
            }
        }
    }

    template <typename T, uint32_t BLOCKDIM>
    static void fill_S_matrix(int      m_pad,
                              int      n,
                              const T* w,
                              const T* v,
                              const T* rhs,
                              T*       S_lower,
                              T*       S_main,
                              T*       S_upper,
                              T*       S_rhs)
    {
        const int S_size = 2 * m_pad / BLOCKDIM;

        // Fill the S matrix using the w and v vectors
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < S_size; i++)
        {
            S_upper[i] = (i % 2 == 0) ? v[i / 2] : static_cast<T>(1);
            S_lower[i]
                = (i % 2 == 0) ? static_cast<T>(1) : w[i / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];

            if(i == 0)
            {
                S_lower[0]          = static_cast<T>(0);
                S_lower[1]          = static_cast<T>(0);
                S_upper[S_size - 2] = static_cast<T>(0);
                S_upper[S_size - 1] = static_cast<T>(0);
                S_main[0]           = static_cast<T>(1);
                S_main[S_size - 1]  = static_cast<T>(1);
            }

            if(i >= 1 && i < S_size - 1)
            {
                S_main[i]
                    = (i % 2 == 0) ? w[i / 2] : v[i / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
            }

            if(i < S_size / 2)
            {
                for(int j = 0; j < n; j++)
                {
                    S_rhs[2 * i + S_size * j] = rhs[i + m_pad * j];
                    S_rhs[2 * i + 1 + S_size * j]
                        = rhs[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1) + m_pad * j];
                }
            }
        }
    }

    template <typename T>
    static void
        S_solve(int m, int n, const T* lower_diag, const T* main_diag, const T* upper_diag, T* rhs)
    {
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
                for(int j = 0; j < n; j++)
                {
                    T rhsk = rhs[k + m * j] * inv_bk;

                    rhs[k + m * j] = rhsk;

                    if(k < (m - 1))
                    {
                        rhs[k + 1 + m * j] += -(ak_1 * rhsk);
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
                for(int j = 0; j < n; j++)
                {
                    T rhsk   = rhs[k + m * j] * det;
                    T rhsk_1 = rhs[k + 1 + m * j] * det;

                    rhs[k + m * j]     = (bk_1 * rhsk - ck * rhsk_1);
                    rhs[k + 1 + m * j] = (-ak_1 * rhsk + bk * rhsk_1);

                    if(k < (m - 2))
                    {
                        rhs[k + 2 + m * j] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);
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

                for(int j = 0; j < n; j++)
                {
                    rhs[k + m * j] += -tmp * rhs[k + 1 + m * j];
                }

                k -= 1;
            }
            else
            {
                const T tmp1 = mt[k];
                const T tmp2 = mt[k - 1];

                for(int j = 0; j < n; j++)
                {
                    rhs[k + m * j] += -tmp1 * rhs[k + 1 + m * j];
                    rhs[k - 1 + m * j] += -tmp2 * rhs[k + 1 + m * j];
                }

                k -= 2;
            }
        }
    }

    template <typename T, uint32_t BLOCKDIM>
    static void backward_solve(int m_pad, int n, const T* w, const T* v, T* rhs)
    {
        const int nblocks = m_pad / BLOCKDIM;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < nblocks; i++)
        {
            double x1 = (i >= 1) ? rhs[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (i - 1)] : 0.0f;
            double x2 = (i < (m_pad / BLOCKDIM - 1)) ? rhs[i + 1] : 0.0f;

            for(int j = 1; j < BLOCKDIM - 1; j++)
            {
                for(int k = 0; k < n; k++)
                {
                    rhs[(m_pad / BLOCKDIM) * j + i + m_pad * k]
                        = rhs[(m_pad / BLOCKDIM) * j + i + m_pad * k]
                          - w[(m_pad / BLOCKDIM) * j + i] * x1 - v[(m_pad / BLOCKDIM) * j + i] * x2;
                }
            }
        }
    }

    template <typename T>
    static void host_spike_algorithm_impl(int                      m,
                                          int                      n,
                                          const T*                 lower_diag,
                                          const T*                 main_diag,
                                          const T*                 upper_diag,
                                          const T*                 B,
                                          T*                       X,
                                          const tridiagonal_descr* descr)
    {
        constexpr int BLOCKDIM = 8;

        const int m_pad = next_power_of_two(m);

        for(size_t i = 0; i < m_pad; i++)
        {
            descr->host_data.lower_pad[i] = static_cast<T>(0);
            descr->host_data.main_pad[i]  = static_cast<T>(1);
            descr->host_data.upper_pad[i] = static_cast<T>(0);

            for(int j = 0; j < n; j++)
            {
                descr->host_data.B_pad[i + m_pad * j] = static_cast<T>(0);
            }
        }

        // Perform data marshalling with padding
        data_marshalling<T, BLOCKDIM>(m,
                                      m_pad,
                                      n,
                                      lower_diag,
                                      main_diag,
                                      upper_diag,
                                      B,
                                      descr->host_data.lower_pad.data(),
                                      descr->host_data.main_pad.data(),
                                      descr->host_data.upper_pad.data(),
                                      descr->host_data.B_pad.data());

        for(int i = 0; i < m_pad; i++)
        {
            descr->host_data.w_pad[i] = static_cast<T>(0);
            descr->host_data.v_pad[i] = static_cast<T>(0);
            descr->host_data.mt[i]    = static_cast<T>(0);
        }

        LBMT_solve<T, BLOCKDIM>(m_pad,
                                n,
                                descr->host_data.lower_pad.data(),
                                descr->host_data.main_pad.data(),
                                descr->host_data.upper_pad.data(),
                                descr->host_data.w_pad.data(),
                                descr->host_data.v_pad.data(),
                                descr->host_data.mt.data(),
                                descr->host_data.B_pad.data());

        const int S_size = 2 * m_pad / BLOCKDIM;

        for(int i = 0; i < S_size; i++)
        {
            descr->host_data.S_lower[i] = static_cast<T>(0);
            descr->host_data.S_main[i]  = static_cast<T>(0);
            descr->host_data.S_upper[i] = static_cast<T>(0);

            for(int j = 0; j < n; j++)
            {
                descr->host_data.S_B[i + S_size * j] = static_cast<T>(0);
            }
        }

        fill_S_matrix<T, BLOCKDIM>(m_pad,
                                   n,
                                   descr->host_data.w_pad.data(),
                                   descr->host_data.v_pad.data(),
                                   descr->host_data.B_pad.data(),
                                   descr->host_data.S_lower.data(),
                                   descr->host_data.S_main.data(),
                                   descr->host_data.S_upper.data(),
                                   descr->host_data.S_B.data());

        S_solve<T>(S_size,
                   n,
                   descr->host_data.S_lower.data(),
                   descr->host_data.S_main.data(),
                   descr->host_data.S_upper.data(),
                   descr->host_data.S_B.data());

        // Write S_B back to B_pad
        for(int j = 0; j < n; j++)
        {
            for(int i = 1; i < S_size - 1; i += 2)
            {
                double temp                              = descr->host_data.S_B[i + S_size * j];
                descr->host_data.S_B[i + S_size * j]     = descr->host_data.S_B[i + 1 + S_size * j];
                descr->host_data.S_B[i + 1 + S_size * j] = temp;
            }

            for(int i = 0; i < S_size / 2; i++)
            {
                descr->host_data.B_pad[i + m_pad * j] = descr->host_data.S_B[2 * i + S_size * j];
                descr->host_data.B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1) + m_pad * j]
                    = descr->host_data.S_B[2 * i + 1 + S_size * j];
            }
        }

        backward_solve<T, BLOCKDIM>(m_pad,
                                    n,
                                    descr->host_data.w_pad.data(),
                                    descr->host_data.v_pad.data(),
                                    descr->host_data.B_pad.data());

        data_marshalling2<T, BLOCKDIM>(m, m_pad, n, descr->host_data.B_pad.data(), X);
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
        // No additional buffers needed for Thomas algorithm
        break;
    }
    case pivoting_strategy::partial:
    {
        const int m_pad = next_power_of_two(m);

        std::cout << "analysis m: " << m << ", m_pad: " << m_pad << std::endl;

        descr->host_data.lower_pad.resize(m_pad, 0.0);
        descr->host_data.main_pad.resize(m_pad, 1.0);
        descr->host_data.upper_pad.resize(m_pad, 0.0);
        descr->host_data.B_pad.resize(m_pad * n, 0.0);

        // For the spike algorithm, we need to allocate buffers for the padded system
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
                                  x.get_vec(),
                                  descr);
        break;
    }
    }
}
