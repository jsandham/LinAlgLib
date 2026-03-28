#include "host_scale.h"

#include "../../trace.h"

namespace linalg
{
    template <typename T>
    static void host_scale_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int n, T scalar)
    {
        ROUTINE_TRACE("host_scale_diagonal_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                if(csr_col_ind[j] == i)
                {
                    csr_val[j] *= scalar;
                    break;
                }
            }
        }
    }

    template <typename T>
    static void host_scale_by_inverse_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int n, const T* diag)
    {
        ROUTINE_TRACE("host_scale_by_inverse_diagonal_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                csr_val[j] *= (1.0 / diag[csr_col_ind[j]]);
            }
        }
    }
}

void linalg::host_scale_diagonal(csr_matrix& A, double scalar)
{
    ROUTINE_TRACE("linalg::host_scale_diagonal");
    host_scale_diagonal_impl(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), scalar);
}

void linalg::host_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag)
{
    ROUTINE_TRACE("linalg::host_scale_by_inverse_diagonal");
    host_scale_by_inverse_diagonal_impl(
        A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), diag.get_vec());
}
