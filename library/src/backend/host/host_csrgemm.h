#ifndef HOST_CSRGEMM_H
#define HOST_CSRGEMM_H

#include "csr_matrix.h"
#include "linalg_enums.h"

namespace linalg
{
    struct csrgemm_descr;

    void host_csrgemm_nnz(const csr_matrix& A,
                          const csr_matrix& B,
                          const csr_matrix& D,
                          csr_matrix&       C,
                          csrgemm_algorithm alg,
                          csrgemm_descr*    descr);
    void host_csrgemm_solve(double               alpha,
                            const csr_matrix&    A,
                            const csr_matrix&    B,
                            double               beta,
                            const csr_matrix&    D,
                            csr_matrix&          C,
                            csrgemm_algorithm    alg,
                            const csrgemm_descr* descr);
}

#endif
