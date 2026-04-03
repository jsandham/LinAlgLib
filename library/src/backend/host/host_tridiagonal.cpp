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

#include "host_tridiagonal.h"

namespace linalg
{
    template <uint32_t M, typename T>
    static void host_thomas_algorithm_impl(
        int n, const T* lower_diag, const T* main_diag, const T* upper_diag, const T* b, T* x)
    {
        ROUTINE_TRACE("host_thomas_algorithm_impl");
        T c_prime[M];
        c_prime[0] = upper_diag[0] / main_diag[0];
        for(int i = 1; i < M - 1; i++)
        {
            T denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
            c_prime[i] = upper_diag[i] / denom;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int j = 0; j < n; j++)
        {
            T d_prime[M];
            d_prime[0] = b[M * j + 0] / main_diag[0];
            for(int i = 1; i < M; i++)
            {
                T num      = b[M * j + i] - lower_diag[i] * d_prime[i - 1];
                T denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
                d_prime[i] = num / denom;
            }
            x[M * j + (M - 1)] = d_prime[M - 1];
            for(int i = M - 2; i >= 0; i--)
                x[M * j + i] = d_prime[i] - c_prime[i] * x[M * j + (i + 1)];
        }
    }
}

void linalg::host_tridiagonal_analysis(int                  m,
                                       int                  n,
                                       const vector<float>& lower_diag,
                                       const vector<float>& main_diag,
                                       const vector<float>& upper_diag,
                                       tridiagonal_descr*   descr)
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
                                     const vector<float>&     lower_diag,
                                     const vector<float>&     main_diag,
                                     const vector<float>&     upper_diag,
                                     const vector<float>&     b,
                                     vector<float>&           x,
                                     const tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::host_tridiagonal_solver");
    assert(main_diag.get_size() == m);
    assert(lower_diag.get_size() == m);
    assert(upper_diag.get_size() == m);
    assert(b.get_size() == m * n);
    assert(x.get_size() == m * n);
    switch(m)
    {
    case 2:
        host_thomas_algorithm_impl<2>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 3:
        host_thomas_algorithm_impl<3>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 4:
        host_thomas_algorithm_impl<4>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 5:
        host_thomas_algorithm_impl<5>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 6:
        host_thomas_algorithm_impl<6>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 7:
        host_thomas_algorithm_impl<7>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 8:
        host_thomas_algorithm_impl<8>(n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      b.get_vec(),
                                      x.get_vec());
        break;
    case 16:
        host_thomas_algorithm_impl<16>(n,
                                       lower_diag.get_vec(),
                                       main_diag.get_vec(),
                                       upper_diag.get_vec(),
                                       b.get_vec(),
                                       x.get_vec());
        break;
    case 32:
        host_thomas_algorithm_impl<32>(n,
                                       lower_diag.get_vec(),
                                       main_diag.get_vec(),
                                       upper_diag.get_vec(),
                                       b.get_vec(),
                                       x.get_vec());
        break;
    case 64:
        host_thomas_algorithm_impl<64>(n,
                                       lower_diag.get_vec(),
                                       main_diag.get_vec(),
                                       upper_diag.get_vec(),
                                       b.get_vec(),
                                       x.get_vec());
        break;
    case 128:
        host_thomas_algorithm_impl<128>(n,
                                        lower_diag.get_vec(),
                                        main_diag.get_vec(),
                                        upper_diag.get_vec(),
                                        b.get_vec(),
                                        x.get_vec());
        break;
    default:
        break;
    }
}
