#ifndef HOST_SCALE_H
#define HOST_SCALE_H

#include "csr_matrix.h"
#include "vector.h"

namespace linalg
{
    void host_scale_diagonal(csr_matrix& A, double scalar);
    void host_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag);
}

#endif
