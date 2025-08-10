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