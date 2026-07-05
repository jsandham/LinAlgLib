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

#include <vector>

#include "../../trace.h"

#include "thomas_algorithm.h"

namespace linalg
{
    template <typename T>
    void thomas_algorithm_template(int      m,
                                   int      n,
                                   const T* lower_diag,
                                   const T* main_diag,
                                   const T* upper_diag,
                                   const T* b,
                                   T*       x)
    {
        ROUTINE_TRACE("thomas_algorithm_template");

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
            {
                x[m * j + i] = d_prime[i] - c_prime[i] * x[m * j + (i + 1)];
            }
        }
    }

    template void thomas_algorithm_template<float>(
        int, int, const float*, const float*, const float*, const float*, float*);
    template void thomas_algorithm_template<double>(
        int, int, const double*, const double*, const double*, const double*, double*);
}
