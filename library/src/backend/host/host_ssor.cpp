#include <cassert>
#include <vector>

#include "../../trace.h"
#include "host_ssor.h"

namespace linalg
{
    template <typename T>
    static void host_diagonal_local(
        const int* csr_row_ptr, const int* csr_col_ind, const T* csr_val, T* d, int n)
    {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                if(csr_col_ind[j] == i)
                {
                    d[i] = csr_val[j];
                    break;
                }
            }
        }
    }

    template <typename T>
    static void host_ssor_fill_lower_precond_impl(int        m_A,
                                                  int        n_A,
                                                  int        nnz_A,
                                                  const int* csr_row_ptr_A,
                                                  const int* csr_col_ind_A,
                                                  const T*   csr_val_A,
                                                  int        m_L,
                                                  int        n_L,
                                                  int        nnz_L,
                                                  const int* csr_row_ptr_L,
                                                  int*       csr_col_ind_L,
                                                  T*         csr_val_L,
                                                  T          omega)
    {
        ROUTINE_TRACE("host_ssor_fill_lower_precond_impl");
        std::vector<T> diag(m_A, 0.0);
        host_diagonal_local(csr_row_ptr_A, csr_col_ind_A, csr_val_A, diag.data(), m_A);
        T beta = 1.0 / omega;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int index = csr_row_ptr_L[i];
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                int col_A = csr_col_ind_A[j];
                if(col_A < i)
                {
                    csr_col_ind_L[index] = col_A;
                    csr_val_L[index]     = csr_val_A[j] / diag[col_A];
                    index++;
                }
                else if(col_A == i)
                {
                    csr_col_ind_L[index] = col_A;
                    csr_val_L[index]     = beta;
                    index++;
                }
            }
            assert(index == csr_row_ptr_L[i + 1]);
        }
    }

    template <typename T>
    static void host_ssor_fill_upper_precond_impl(int        m_A,
                                                  int        n_A,
                                                  int        nnz_A,
                                                  const int* csr_row_ptr_A,
                                                  const int* csr_col_ind_A,
                                                  const T*   csr_val_A,
                                                  int        m_U,
                                                  int        n_U,
                                                  int        nnz_U,
                                                  const int* csr_row_ptr_U,
                                                  int*       csr_col_ind_U,
                                                  T*         csr_val_U,
                                                  T          omega)
    {
        ROUTINE_TRACE("host_ssor_fill_upper_precond_impl");
        T beta = 1.0 / omega;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int index = csr_row_ptr_U[i];
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                int col_A = csr_col_ind_A[j];
                if(col_A > i)
                {
                    csr_col_ind_U[index] = col_A;
                    csr_val_U[index]     = csr_val_A[j];
                    index++;
                }
                else if(col_A == i)
                {
                    csr_col_ind_U[index] = col_A;
                    csr_val_U[index]     = beta * csr_val_A[j];
                    index++;
                }
            }
            assert(index == csr_row_ptr_U[i + 1]);
        }
    }
}

void linalg::host_ssor_fill_lower_precond(const csr_matrix& A, csr_matrix& L, double omega)
{
    ROUTINE_TRACE("linalg::host_ssor_fill_lower_precond");
    host_ssor_fill_lower_precond_impl(A.get_m(),
                                      A.get_n(),
                                      A.get_nnz(),
                                      A.get_row_ptr(),
                                      A.get_col_ind(),
                                      A.get_val(),
                                      L.get_m(),
                                      L.get_n(),
                                      L.get_nnz(),
                                      L.get_row_ptr(),
                                      L.get_col_ind(),
                                      L.get_val(),
                                      omega);
}

void linalg::host_ssor_fill_upper_precond(const csr_matrix& A, csr_matrix& U, double omega)
{
    ROUTINE_TRACE("linalg::host_ssor_fill_upper_precond");
    host_ssor_fill_upper_precond_impl(A.get_m(),
                                      A.get_n(),
                                      A.get_nnz(),
                                      A.get_row_ptr(),
                                      A.get_col_ind(),
                                      A.get_val(),
                                      U.get_m(),
                                      U.get_n(),
                                      U.get_nnz(),
                                      U.get_row_ptr(),
                                      U.get_col_ind(),
                                      U.get_val(),
                                      omega);
}
