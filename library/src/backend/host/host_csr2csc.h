#ifndef HOST_CSR2CSC_H
#define HOST_CSR2CSC_H

#include "csr_matrix.h"

namespace linalg
{
    void host_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA);
}

#endif
