//********************************************************************************
#ifndef DEVICE_CSRIC0_H
#define DEVICE_CSRIC0_H

#include "csr_matrix.h"

namespace linalg
{
    void device_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero);

    struct csric0_descr;

    void allocate_csric0_device_data(csric0_descr* descr);
    void free_csric0_device_data(csric0_descr* descr);
    void device_csric0_analysis(const csr_matrix& A, csric0_descr* descr);
    void device_csric0_compute(csr_matrix& A, const csric0_descr* descr);
}

#endif
