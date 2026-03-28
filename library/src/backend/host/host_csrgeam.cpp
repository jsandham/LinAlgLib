#include <algorithm>
#include <iostream>
#include <vector>

#include "../../trace.h"
#include "host_csrgeam.h"

namespace linalg
{
    template <typename T>
    static void host_csrgeam_nnz_impl(int        m,
                                      int        n,
                                      int        nnz_A,
                                      int        nnz_B,
                                      T          alpha,
                                      const int* csr_row_ptr_A,
                                      const int* csr_col_ind_A,
                                      T          beta,
                                      const int* csr_row_ptr_B,
                                      const int* csr_col_ind_B,
                                      int*       csr_row_ptr_C,
                                      int*       nnz_C)
    {
        ROUTINE_TRACE("host_csrgeam_nnz_impl");
        csr_row_ptr_C[0] = 0;
        for(int i = 0; i < m; i++)
        {
            std::vector<int> nnz(n, -1);
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
                nnz[csr_col_ind_A[j]] = 1;
            for(int j = csr_row_ptr_B[i]; j < csr_row_ptr_B[i + 1]; j++)
                nnz[csr_col_ind_B[j]] = 1;
            int row_nnz = 0;
            for(int j = 0; j < n; j++)
                if(nnz[j] != -1)
                    row_nnz++;
            csr_row_ptr_C[i + 1] = row_nnz;
        }
        for(int i = 0; i < m; i++)
            csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
        *nnz_C = csr_row_ptr_C[m];
    }

    template <typename T>
    static void host_csrgeam_impl(int        m,
                                  int        n,
                                  int        nnz_A,
                                  int        nnz_B,
                                  T          alpha,
                                  const int* csr_row_ptr_A,
                                  const int* csr_col_ind_A,
                                  const T*   csr_val_A,
                                  T          beta,
                                  const int* csr_row_ptr_B,
                                  const int* csr_col_ind_B,
                                  const T*   csr_val_B,
                                  const int* csr_row_ptr_C,
                                  int*       csr_col_ind_C,
                                  T*         csr_val_C)
    {
        ROUTINE_TRACE("host_csrgeam_impl");
        for(int i = 0; i < m; i++)
        {
            std::vector<int> nnz(n, -1);
            int              row_begin_C = csr_row_ptr_C[i];
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                csr_col_ind_C[row_begin_C] = csr_col_ind_A[j];
                csr_val_C[row_begin_C]     = alpha * csr_val_A[j];
                nnz[csr_col_ind_A[j]]      = row_begin_C++;
            }
            for(int j = csr_row_ptr_B[i]; j < csr_row_ptr_B[i + 1]; j++)
            {
                int col_B = csr_col_ind_B[j];
                if(nnz[col_B] != -1)
                    csr_val_C[nnz[col_B]] += beta * csr_val_B[j];
                else
                {
                    csr_col_ind_C[row_begin_C] = col_B;
                    csr_val_C[row_begin_C]     = beta * csr_val_B[j];
                    nnz[col_B]                 = row_begin_C++;
                }
            }
        }
        for(int i = 0; i < m; ++i)
        {
            int              row_begin_C = csr_row_ptr_C[i];
            int              row_end_C   = csr_row_ptr_C[i + 1];
            int              row_nnz     = row_end_C - row_begin_C;
            std::vector<int> perm(row_nnz), columns(row_nnz);
            std::vector<T>   values(row_nnz);
            for(int j = 0; j < row_nnz; ++j)
            {
                perm[j]    = j;
                columns[j] = csr_col_ind_C[row_begin_C + j];
                values[j]  = csr_val_C[row_begin_C + j];
            }
            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                return columns[a] < columns[b];
            });
            for(int j = 0; j < row_nnz; ++j)
            {
                csr_col_ind_C[row_begin_C + j] = columns[perm[j]];
                csr_val_C[row_begin_C + j]     = values[perm[j]];
            }
        }
    }
}

void linalg::host_csrgeam_nnz(const csr_matrix& A,
                              const csr_matrix& B,
                              csr_matrix&       C,
                              csrgeam_algorithm alg,
                              csrgeam_descr*    descr)
{
    ROUTINE_TRACE("linalg::host_csrgeam_nnz");
    C.resize(A.get_m(), B.get_n(), 0);
    int nnz_C;
    host_csrgeam_nnz_impl(A.get_m(),
                          B.get_n(),
                          A.get_nnz(),
                          B.get_nnz(),
                          1.0,
                          A.get_row_ptr(),
                          A.get_col_ind(),
                          1.0,
                          B.get_row_ptr(),
                          B.get_col_ind(),
                          C.get_row_ptr(),
                          &nnz_C);
    C.resize(A.get_m(), B.get_n(), nnz_C);
}

void linalg::host_csrgeam_solve(double               alpha,
                                const csr_matrix&    A,
                                double               beta,
                                const csr_matrix&    B,
                                csr_matrix&          C,
                                csrgeam_algorithm    alg,
                                const csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csrgeam_solve");
    host_csrgeam_impl(A.get_m(),
                      B.get_n(),
                      A.get_nnz(),
                      B.get_nnz(),
                      alpha,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      beta,
                      B.get_row_ptr(),
                      B.get_col_ind(),
                      B.get_val(),
                      C.get_row_ptr(),
                      C.get_col_ind(),
                      C.get_val());
}
