#ifndef HOST_EXTRACT_H
#define HOST_EXTRACT_H

#include "csr_matrix.h"
#include "vector.h"

namespace linalg
{
    void host_diagonal(const csr_matrix& A, vector<double>& d);
    void host_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L);
    void host_extract_lower_triangular(const csr_matrix& A, csr_matrix& L);
    void host_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U);
    void host_extract_upper_triangular(const csr_matrix& A, csr_matrix& U);
}

#endif
