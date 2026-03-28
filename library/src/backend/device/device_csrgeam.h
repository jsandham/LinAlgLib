//********************************************************************************
#ifndef DEVICE_CSRGEAM_H
#define DEVICE_CSRGEAM_H

#include "csr_matrix.h"
#include "linalg_enums.h"

namespace linalg
{
    struct csrgeam_descr;

    void allocate_csrgeam_device_data(csrgeam_descr* descr);
    void free_csrgeam_device_data(csrgeam_descr* descr);
    void device_csrgeam_nnz(const csr_matrix& A,
                            const csr_matrix& B,
                            csr_matrix&       C,
                            csrgeam_algorithm alg,
                            csrgeam_descr*    descr);
    void device_csrgeam_solve(double               alpha,
                              const csr_matrix&    A,
                              double               beta,
                              const csr_matrix&    B,
                              csr_matrix&          C,
                              csrgeam_algorithm    alg,
                              const csrgeam_descr* descr);
}

#endif
