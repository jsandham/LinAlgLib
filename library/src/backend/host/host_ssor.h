#ifndef HOST_SSOR_H
#define HOST_SSOR_H

#include "csr_matrix.h"

namespace linalg
{
    void host_ssor_fill_lower_precond(const csr_matrix& A, csr_matrix& L, double omega);
    void host_ssor_fill_upper_precond(const csr_matrix& A, csr_matrix& U, double omega);
}

#endif
