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

#include "../../csr_matrix.h"
#include "../../linalg_export.h"

#include "amg_util.h"

/*! \file
 *  \brief amg_strength.h provides interface for computing strong connections in
 * a matrix. Used algebraic multigrid solvers
 */

/*! \addtogroup iterative_solvers_amg
 *  @{ */

namespace linalg
{
    /*! \ingroup iterative_solvers_amg
 *  \brief Compute smoothed-aggregation strong connections.
 *
 *  \details
 *  Determines which graph edges are considered strong for aggregation-based
 *  coarsening. The output connection vector is then used by aggregation
 *  routines to form coarse-grid aggregates.
 *
 *  \param A The system matrix whose connectivity graph is analyzed.
 *  \param eps Strength threshold used to decide whether a connection is strong.
 *  \param connections Output vector encoding the detected strong connections.
 */
    LINALGLIB_API void
        compute_strong_connections(const csr_matrix& A, double eps, vector<int>& connections);

    /*! \ingroup iterative_solvers_amg
 *  \brief Compute the classical AMG strength matrix.
 *
 *  \details
 *  Builds the classical strength-of-connection matrix and, optionally, a
 *  companion connection vector used by classical AMG coarsening/interpolation.
 *
 *  \param A The system matrix whose strong couplings are analyzed.
 *  \param theta Classical AMG strength threshold parameter.
 *  \param S Output CSR matrix storing the strength-of-connection pattern.
 *  \param connections Output vector describing strong connections per node.
 */
    LINALGLIB_API void compute_classical_strong_connections(const csr_matrix& A,
                                                            double            theta,
                                                            csr_matrix&       S,
                                                            vector<int>&      connections);
}

/*! @} */

#endif
