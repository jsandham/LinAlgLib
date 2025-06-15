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

#ifndef AMG_AGGREGATION_H
#define AMG_AGGREGATION_H

#include <cstdint>
#include <vector>

#include "../../linalg_export.h"
#include "../../csr_matrix.h"

#include "amg_util.h"

/*! \file
 *  \brief amg_aggregation.h provides interface for creating aggregations used
 *  algebraic multigrid solvers
 */

namespace linalg
{
/*! \ingroup iterative_solvers
 *  \brief Compute aggregates using parallel maximum independent set
 */
LINALGLIB_API bool compute_aggregates_using_pmis(const csr_matrix &A, const vector<int> &connections,
                                   vector<int64_t> &aggregates, vector<int64_t> &aggregate_root_nodes);

/*! \ingroup iterative_solvers
 *  \brief Compute classical C/F points (first pass)
 */
LINALGLIB_API void compute_cfpoint_first_pass(const csr_matrix &S, const csr_matrix &ST, vector<uint32_t> &cfpoints);

 /*! \ingroup iterative_solvers
  *  \brief Compute classical C/F points (second pass)
  */
LINALGLIB_API void compute_cfpoint_second_pass(const csr_matrix &S, vector<uint32_t> &cfpoints);
}
 

#endif