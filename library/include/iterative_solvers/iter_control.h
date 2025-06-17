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

#ifndef ITER_CONTROL_H
#define ITER_CONTROL_H

/*! \file
 *  \brief iter_control.h provides a structure used for controling the number of iterations 
 *  in Classical, krylov, and amg solvers
 */

namespace linalg
{
/**
 * @brief Structure to control the iterative solution process.
 *
 * This struct provides parameters and methods to determine the convergence
 * and termination of an iterative algorithm.
 */
struct iter_control
{
    /**
    * @brief Relative tolerance for convergence.
    *
    * The iteration is considered converged if the relative reduction
    * in the residual norm is less than or equal to this value.
    * Defaults to 1e-06.
    */
    double rel_tol = 1e-06;

    /**
    * @brief Absolute tolerance for convergence.
    *
    * The iteration is considered converged if the absolute value
    * of the residual norm is less than or equal to this value.
    * Defaults to 1e-06.
    */
    double abs_tol = 1e-06;

    /**
    * @brief Maximum number of iterations allowed.
    *
    * The iterative process will terminate if the number of iterations
    * exceeds this value. Defaults to 1000.
    */
    int max_iter = 1000;

    /**
    * @brief Maximum number of cycles allowed (for iterative methods with inner loops).
    *
    * For iterative methods that involve cycles or inner loops, the process
    * will terminate if the number of cycles exceeds this value.
    * Defaults to 200.
    */
    int max_cycle = 200;

    /**
    * @brief Checks if the residual norm has converged based on relative and absolute tolerances.
    *
    * The convergence criterion is met if either the relative reduction
    * in the residual norm is less than or equal to \ref rel_tol, OR
    * the absolute value of the residual norm is less than or equal to \ref abs_tol.
    *
    * @param residual_norm The current residual norm.
    * @param initial_residual_norm The initial residual norm at the start of the iteration.
    * @return True if the residual has converged, false otherwise.
    */
    bool residual_converges(double residual_norm, double initial_residual_norm) const;

    /**
    * @brief Checks if the maximum number of iterations has been exceeded.
    *
    * @param iter The current iteration number.
    * @return True if the current iteration exceeds \ref max_iter, false otherwise.
    */
    bool exceed_max_iter(int iter) const;

    /**
    * @brief Checks if the maximum number of cycles has been exceeded.
    *
    * @param cycle The current cycle number.
    * @return True if the current cycle exceeds \ref max_cycle, false otherwise.
    */
    bool exceed_max_cycle(int cycle) const;
};
}

#endif