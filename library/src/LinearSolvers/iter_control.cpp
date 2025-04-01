#include "../../include/LinearSolvers/iter_control.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cmath>

#define DEBUG 1

bool iter_control::residual_converges(double residual_norm, double initial_residual_norm) const
{
#if (DEBUG)
    std::cout << "absolute residual: " << residual_norm << " relative residual: " << residual_norm / initial_residual_norm << std::endl;
#endif

    assert(rel_tol > 0.0);
    assert(abs_tol > 0.0);

    if(std::isnan(residual_norm))
    {
        return false;
    }
    if(std::isnan(initial_residual_norm))
    {
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