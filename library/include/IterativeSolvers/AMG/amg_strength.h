//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#ifndef AMG_STRENGTH_H
#define AMG_STRENGTH_H

#include <vector>

#include "../../linalglib_export.h"
#include "../../csr_matrix.h"

#include "amg_util.h"

/*! \file
 *  \brief amg_strength.h provides interface for computing strong connections in
 * a matrix. Used algebraic multigrid solvers
 */

/*! \ingroup iterative_solvers
 *  \brief Compute smoothed aggregation strong connections
 */
LINALGLIB_API void compute_strong_connections(const csr_matrix &A, double eps, std::vector<int> &connections);
LINALGLIB_API void compute_strong_connections(const csr_matrix2 &A, double eps, std::vector<int> &connections);

/*! \ingroup iterative_solvers
 *  \brief Compute classical strength matrix
 */
LINALGLIB_API void compute_classical_strong_connections(const csr_matrix &A, double theta, csr_matrix &S, std::vector<int> &connections);
LINALGLIB_API void compute_classical_strong_connections(const csr_matrix2 &A, double theta, csr_matrix2 &S, std::vector<int> &connections);

#endif