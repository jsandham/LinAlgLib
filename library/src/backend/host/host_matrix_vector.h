#ifndef HOST_MATRIX_VECTOR_H
#define HOST_MATRIX_VECTOR_H

#include "csr_matrix.h"
#include "linalg_enums.h"
#include "vector.h"

namespace linalg
{
    void host_compute_residual(const csr_matrix&     A,
                               const vector<double>& x,
                               const vector<double>& b,
                               vector<double>&       res);

    struct csrmv_descr;

    void host_csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr);
    void host_csrmv_solve(double                alpha,
                          const csr_matrix&     A,
                          const vector<double>& x,
                          double                beta,
                          vector<double>&       y,
                          csrmv_algorithm       alg,
                          const csrmv_descr*    descr);
}

#endif
