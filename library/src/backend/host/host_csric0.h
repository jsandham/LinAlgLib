#ifndef HOST_CSRIC0_H
#define HOST_CSRIC0_H

#include "csr_matrix.h"

namespace linalg
{
    void host_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero);

    struct csric0_descr;

    void host_csric0_analysis(const csr_matrix& A, csric0_descr* descr);
    void host_csric0_compute(csr_matrix& A, const csric0_descr* descr);
}

#endif
