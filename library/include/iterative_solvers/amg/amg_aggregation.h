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

#include "../../csr_matrix.h"
#include "../../linalg_export.h"

#include "amg_util.h"

/*! \file
 *  \brief amg_aggregation.h provides interface for creating aggregations used
 *  algebraic multigrid solvers
 */

/*! \addtogroup iterative_solvers_amg
 *  @{ */

namespace linalg
{
    /*! \ingroup iterative_solvers_amg
 *  \brief Compute aggregates using a parallel maximal independent set strategy.
 *
 *  \details
 *  Builds aggregate assignments for AMG coarsening by selecting aggregate root
 *  nodes and assigning fine-grid unknowns to those aggregates. The routine uses
 *  the supplied strength-of-connection information to determine which nodes can
 *  be grouped together.
 *
 *  \param A The system matrix whose graph defines the aggregation problem.
 *  \param connections A vector describing strong connections between nodes.
 *  \param aggregates Output aggregate ids for each node. On return, each entry
 *  identifies the aggregate to which the corresponding node belongs.
 *  \param aggregate_root_nodes Output marker vector identifying aggregate root
 *  nodes used to seed the aggregates.
 *  \return `true` if aggregation completed successfully, `false` if the input
 *  data are on mixed backends or aggregation failed.
 */
    LINALGLIB_API bool compute_aggregates_using_pmis(const csr_matrix&  A,
                                                     const vector<int>& connections,
                                                     vector<int64_t>&   aggregates,
                                                     vector<int64_t>&   aggregate_root_nodes);

    /*! \ingroup iterative_solvers_amg
 *  \brief Compute classical C/F point labels for the first coarsening pass.
 *
 *  \details
 *  Performs the initial Ruge-Stuben style classification of nodes into coarse
 *  and fine candidates using the strength matrix and its transpose.
 *
 *  \param S The strength-of-connection matrix.
 *  \param ST The transpose of the strength-of-connection matrix.
 *  \param cfpoints Output vector of point labels. On return, entries are marked
 *  as coarse, fine, or intermediate states produced by the first pass.
 */
    LINALGLIB_API void compute_cfpoint_first_pass(const csr_matrix& S,
                                                  const csr_matrix& ST,
                                                  vector<uint32_t>& cfpoints);

    /*! \ingroup iterative_solvers_amg
  *  \brief Finalize classical C/F point labels in the second coarsening pass.
  *
  *  \details
  *  Refines the point classification produced by compute_cfpoint_first_pass()
  *  so that remaining undecided nodes are assigned consistently for classical
  *  AMG interpolation.
  *
  *  \param S The strength-of-connection matrix.
  *  \param cfpoints Input/output vector of point labels. On input it contains
  *  the first-pass classification; on output it contains the finalized C/F
  *  labeling.
  */
    LINALGLIB_API void compute_cfpoint_second_pass(const csr_matrix& S, vector<uint32_t>& cfpoints);
}

/*! @} */

#endif
