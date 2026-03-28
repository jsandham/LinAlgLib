//********************************************************************************
#ifndef DEVICE_CSRGEMM_H
#define DEVICE_CSRGEMM_H

#include "csr_matrix.h"
#include "linalg_enums.h"

namespace linalg
{
    struct csrgemm_descr;

    void allocate_csrgemm_device_data(csrgemm_descr* descr);
    void free_csrgemm_device_data(csrgemm_descr* descr);
    void device_csrgemm_nnz(const csr_matrix& A,
                            const csr_matrix& B,
                            const csr_matrix& D,
                            csr_matrix&       C,
                            csrgemm_algorithm alg,
                            csrgemm_descr*    descr);
    void device_csrgemm_solve(double               alpha,
                              const csr_matrix&    A,
                              const csr_matrix&    B,
                              double               beta,
                              const csr_matrix&    D,
                              csr_matrix&          C,
                              csrgemm_algorithm    alg,
                              const csrgemm_descr* descr);
}

#endif
