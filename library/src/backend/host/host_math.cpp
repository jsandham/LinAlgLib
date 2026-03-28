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

#include <algorithm>
#include <cmath>

#include "../../trace.h"
#include "host_math.h"

namespace linalg
{
    template <typename T>
    static T host_norm_inf_impl(const T* array, int n)
    {
        ROUTINE_TRACE("host_norm_inf_impl");

        T norm = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(max : norm)
#endif
        for(int i = 0; i < n; i++)
        {
            norm = std::max(std::abs(array[i]), norm);
        }

        return norm;
    }

    template <typename T>
    static void host_jacobi_solve_impl(const T* rhs, const T* diag, T* x, size_t n)
    {
        ROUTINE_TRACE("host_jacobi_solve_impl");

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < n; i++)
        {
            x[i] = rhs[i] / diag[i];
        }
    }
}

double linalg::host_norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::host_norm_euclid");

    return std::sqrt(host_dot_product(array, array));
}

double linalg::host_norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::host_norm_inf");

    return host_norm_inf_impl(array.get_vec(), array.get_size());
}

void linalg::host_jacobi_solve(const vector<double>& rhs,
                               const vector<double>& diag,
                               vector<double>&       x)
{
    ROUTINE_TRACE("linalg::host_jacobi_solve");

    host_jacobi_solve_impl(rhs.get_vec(), diag.get_vec(), x.get_vec(), x.get_size());
}
