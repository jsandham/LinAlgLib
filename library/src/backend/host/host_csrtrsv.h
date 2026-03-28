#ifndef HOST_CSRTRSV_H
#define HOST_CSRTRSV_H

#include "csr_matrix.h"
#include "linalg_enums.h"
#include "vector.h"

namespace linalg
{
    struct csrtrsv_descr;

    void host_csrtrsv_analysis(const csr_matrix& A,
                               triangular_type   tri_type,
                               diagonal_type     diag_type,
                               csrtrsv_descr*    descr);
    void host_csrtrsv_solve(const csr_matrix&     A,
                            const vector<double>& b,
                            vector<double>&       x,
                            double                alpha,
                            triangular_type       tri_type,
                            diagonal_type         diag_type,
                            const csrtrsv_descr*  descr);
}

#endif
