//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#ifndef LINALG_H__
#define LINALG_H__

/*! \file
*  \brief linalg.h includes other *.h files and provides sparse iterative linear solvers and eigenvalue solvers
*/

/**
 * @mainpage Iterative Solvers Library
 *
 * @section intro_sec Introduction
 *
 * This library provides a comprehensive collection of iterative solvers for large linear systems of the form \f$\mathbf{Ax} = \mathbf{b}\f$.
 * It includes classical stationary iterative methods, Krylov subspace methods, and Algebraic Multigrid (AMG) techniques.
 * This documentation will guide you through the library's structure, how to get started, and the fundamental concepts behind the implemented solvers.
 *
 * @section toc Table of Contents
 *
 * - @ref getting_started
 * - @ref compiling_library
 * - @ref building_documentation
 * - @ref classical_solvers
 * - @ref krylov_solvers
 * - @ref amg_solvers
 * - @ref concepts
 *
 * @section getting_started Getting Started
 *
 * To begin using the Iterative Solvers Library, you'll need to include the linalg.h header file in your C++ project:
 *
 * ```cpp
 * #include "linalg.h"
 * // ... your code ...
 * ```
 *
 * You will typically need to provide the following:
 * - A matrix representing the system (e.g., using a suitable matrix library or your own implementation).
 * - A right-hand side vector \f$\mathbf{b}\f$.
 * - An initial guess for the solution vector \f$\mathbf{x}\f$.
 * - Optionally, parameters such as tolerance for convergence and maximum number of iterations.
 *
 * Example usage (conceptual):
 *
 * ```cpp
 * //  4 -1  0  0 -1
 * // -1  4 -1  0  0
 * //  0 -1  4 -1  0
 * //  0  0 -1  4 -1
 * // -1  0  0 -1  4
 * std::vector<int> csr_row_ptr = {0, 3, 6, 9, 12, 15};
 * std::vector<int> csr_col_ind = {0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4};
 * std::vector<double> csr_val = {4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0};
 *
 * // Solution vector
 * std::vector<double> x(m, 0.0);
 *
 * // Righthand side vector
 * std::vector<double> b(m, 1.0);
 *
 * iter_control control;
 * control.max_iter = 1000;
 * control.rel_tol = 1e-08;
 * control.abs_tol = 1e-08;
 *
 * int iter = jacobi(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, control);
 * ```
 *
 * Please refer to the specific function documentation for detailed usage and available parameters for each solver.
 *
 * @subsection compiling_library Compiling the Library
 *
 * The compilation process for this library will depend on your specific system and build tools.
 * Typically, you will use a build system like CMake.
 *
 * **Using CMake (Recommended):**
 *
 * 1. Create a `build` directory in the root of the library source.
 * ```bash
 * cd LinAlgLib
 * mkdir build
 * cd build
 * ```
 * 2. Configure the build using CMake and build the library
 * ```bash
 * cmake ../
 * cmake --build . --config Debug
 * ```
 *
 * Ensure you have CMake installed on your system. You might also need a C++ compiler (like g++ or clang++).
 * The `CMakeLists.txt` file in the library's root directory contains the necessary build instructions.
 *
 * Refer to the library's specific build instructions for any dependencies or special compilation flags.
 *
 * @subsection building_documentation Building the Documentation
 *
 * This documentation is generated using Doxygen. To build it yourself, you need to have Doxygen installed on your system.
 *
 * 1. Navigate to the root directory of the library.
 * ```bash
 * cd LinAlgLib
 * ``` 
 * 2. Run Doxygen using the provided configuration file (named `Doxyfile`).
 * ```bash
 * doxygen Doxyfile
 * ```
 *
 * This will generate the documentation in the directory specified in the `OUTPUT_DIRECTORY` setting of the `Doxyfile` (usually `doc` or `documentation`). You can then open the `index.html` file in your web browser to view the documentation.
 *
 * @section classical_solvers Classical Stationary Iterative Methods
 *
 * This module contains the following classical iterative solvers:
 * - \ref jacobi : The Jacobi method.
 * - \ref gs : The Gauss-Seidel method.
 * - \ref sor : The Successive Over-Relaxation (SOR) method.
 * - \ref ssor : The Symmetric Successive Over-Relaxation (SSOR) method.
 * - \ref sgs : The Symmetric Gauss-Seidel method.
 * - \ref rich : The Richardson iteration method.
 *
 * These methods are characterized by their simplicity and iterative refinement of the solution based on splitting the system matrix.
 *
 * @section krylov_solvers Krylov Subspace Methods
 *
 * This module implements powerful Krylov subspace methods, which often exhibit faster convergence for large-scale problems:
 * - \ref cg : The Conjugate Gradient (CG) method (suitable for symmetric positive-definite matrices).
 * - \ref bicgstab : The Bi-Conjugate Gradient Stabilized (BiCGSTAB) method (for non-symmetric matrices).
 * - \ref gmres : The Generalized Minimal Residual (GMRES) method (for non-symmetric matrices).
 *
 * These methods build an orthonormal basis for the Krylov subspace and find the approximate solution within that subspace that minimizes the residual.
 *
 * @section amg_solvers Algebraic Multigrid (AMG) Solvers
 *
 * This module features Algebraic Multigrid solvers, which are highly effective for solving large sparse linear systems arising from discretized partial differential equations:
 * - \ref uaamg_setup : AMG based on unsmoothed aggregation.
 * - \ref saamg_setup : AMG based on smoothed aggregation.
 * - \ref rsamg_setup : The classical Ruge-Stüben AMG algorithm.
 *
 * AMG methods use a hierarchy of coarser grids to accelerate the convergence of the iterative process. They involve inter-grid transfer operators (restriction and prolongation) and smoothing operations on each level.
 *
 * @section concepts Fundamental Concepts
 *
 * Understanding the following concepts will be beneficial when using this library:
 *
 * - **Linear System:** A set of linear equations represented in matrix form as \f$\mathbf{Ax} = \mathbf{b}\f$, where \f$\mathbf{A}\f$ is the coefficient matrix, \f$\mathbf{x}\f$ is the unknown solution vector, and \f$\mathbf{b}\f$ is the right-hand side vector.
 * - **Iteration:** An iterative method starts with an initial guess for the solution and refines it in a sequence of steps until a desired level of accuracy is reached.
 * - **Convergence:** An iterative method converges if the sequence of approximate solutions approaches the true solution as the number of iterations increases.
 * - **Residual:** The residual vector \f$\mathbf{r} = \mathbf{b} - \mathbf{Ax}\f$ measures the error in the current approximate solution \f$\mathbf{x}\f$. The goal of an iterative solver is to reduce the norm of the residual.
 * - **Splitting:** Classical iterative methods often rely on splitting the matrix \f$\mathbf{A}\f$ into components (e.g., \f$\mathbf{A} = \mathbf{D} - \mathbf{L} - \mathbf{U}\f$, where \f$\mathbf{D}\f$ is the diagonal, \f$\mathbf{L}\f$ is the lower triangular, and \f$\mathbf{U}\f$ is the upper triangular part).
 * - **Krylov Subspace:** For a matrix \f$\mathbf{A}\f$ and a starting vector \f$\mathbf{r}_0\f$, the \f$k\f$-th Krylov subspace is defined as \f$K_k(\mathbf{A}, \mathbf{r}_0) = \text{span}\{\mathbf{r}_0, \mathbf{Ar}_0, \mathbf{A}^2\mathbf{r}_0, ..., \mathbf{A}^{k-1}\mathbf{r}_0\}\f$. Krylov subspace methods seek an approximate solution in these subspaces.
 * - **Preconditioning:** Preconditioning involves transforming the original linear system into an equivalent system that is easier to solve by an iterative method, often by reducing the condition number of the matrix. This library may include options for preconditioning in some solvers.
 * - **Grid Hierarchy (Multigrid):** AMG methods utilize a hierarchy of coarser grids to accelerate convergence by efficiently reducing errors at different frequency ranges.
 * - **Restriction and Prolongation:** These are operators used in multigrid methods to transfer vectors between different grid levels (coarsening and refinement).
 * - **Smoothing:** Smoothing operations (often a few iterations of a basic iterative method like Gauss-Seidel or Jacobi) are applied on each grid level to reduce high-frequency errors.
 */

// Vector
#include "vector.h"

// CSR matrix
#include "csr_matrix.h"

// Classic Linear solvers
#include "IterativeSolvers/Classic/jacobi.h"
#include "IterativeSolvers/Classic/gauss_seidel.h"
#include "IterativeSolvers/Classic/sor.h"
#include "IterativeSolvers/Classic/symmetric_gauss_seidel.h"
#include "IterativeSolvers/Classic/ssor.h"
#include "IterativeSolvers/Classic/richardson.h"

// Krylov Linear solvers
#include "IterativeSolvers/Krylov/gmres.h"
#include "IterativeSolvers/Krylov/cg.h"
#include "IterativeSolvers/Krylov/bicgstab.h"

// Algrbraic multi-grid solvers
#include "IterativeSolvers/AMG/amg_aggregation.h"
#include "IterativeSolvers/AMG/amg_strength.h"
#include "IterativeSolvers/AMG/amg.h"
#include "IterativeSolvers/AMG/rsamg.h"
#include "IterativeSolvers/AMG/rsamg_old.h"
#include "IterativeSolvers/AMG/saamg.h"
#include "IterativeSolvers/AMG/uaamg.h"

// Preconditioners
#include "IterativeSolvers/Preconditioner/preconditioner.h"

// Eigenvalues solvers
#include "EigenValueSolvers/power_iteration.h"

#include "slaf.h"
#include "IterativeSolvers/iter_control.h"

#endif
