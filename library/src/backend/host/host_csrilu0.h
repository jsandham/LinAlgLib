#ifndef HOST_CSRILU0_H
#define HOST_CSRILU0_H

#include "csr_matrix.h"

namespace linalg
{
    void host_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero);

    struct csrilu0_descr;

    void host_csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr);
    void host_csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr);
}

#endif
