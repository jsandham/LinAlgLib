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

#include "../../include/iterative_solvers/iter_control.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>

#include "../trace.h"

using namespace linalg;

bool iter_control::residual_converges(double residual_norm, double initial_residual_norm) const
{
    ROUTINE_TRACE("residual_converges");

#ifdef CONVERGENCE_LOGGING
    std::cout << "absolute residual: " << residual_norm
              << " relative residual: " << residual_norm / initial_residual_norm << std::endl;
#endif

    assert(rel_tol > 0.0);
    assert(abs_tol > 0.0);

    if(std::isnan(residual_norm))
    {
        std::cout << "Error: NaN detected in residual norm" << std::endl;
        return false;
    }
    if(std::isnan(initial_residual_norm))
    {
        std::cout << "Error: NaN detected in initial residual norm" << std::endl;
        return false;
    }

    assert(!std::isnan(residual_norm));
    assert(!std::isnan(initial_residual_norm));

    assert(residual_norm >= 0.0);
    assert(initial_residual_norm >= 0.0);

    if(residual_norm <= abs_tol)
    {
        return true;
    }

    if(residual_norm / initial_residual_norm <= rel_tol)
    {
        return true;
    }

    return false;
}

bool iter_control::exceed_max_iter(int iter) const
{
    assert(iter >= 0);
    assert(max_iter >= 0);

    return iter > max_iter;
}

bool iter_control::exceed_max_cycle(int cycle) const
{
    return cycle > max_cycle;
}
