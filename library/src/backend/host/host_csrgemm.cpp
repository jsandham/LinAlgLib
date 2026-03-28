#include <algorithm>
#include <cstring>
#include <vector>

#include "../../trace.h"
#include "host_csrgemm.h"

namespace linalg
{
    template <typename T>
    static void host_csrgemm_nnz_impl(int        m,
                                      int        n,
                                      int        k,
                                      int        nnz_A,
                                      int        nnz_B,
                                      int        nnz_D,
                                      T          alpha,
                                      const int* csr_row_ptr_A,
                                      const int* csr_col_ind_A,
                                      const int* csr_row_ptr_B,
                                      const int* csr_col_ind_B,
                                      T          beta,
                                      const int* csr_row_ptr_D,
                                      const int* csr_col_ind_D,
                                      int*       csr_row_ptr_C,
                                      int*       nnz_C)
    {
        ROUTINE_TRACE("host_csrgemm_nnz");
        std::vector<int> nnz(n, -1);
        for(int i = 0; i < m + 1; i++)
            csr_row_ptr_C[i] = 0;
        for(int i = 0; i < m; ++i)
        {
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                int col_A = csr_col_ind_A[j];
                for(int p = csr_row_ptr_B[col_A]; p < csr_row_ptr_B[col_A + 1]; p++)
                {
                    int col_B = csr_col_ind_B[p];
                    if(nnz[col_B] != i)
                    {
                        nnz[col_B] = i;
                        csr_row_ptr_C[i + 1]++;
                    }
                }
            }
            if(beta != 0.0)
            {
                for(int j = csr_row_ptr_D[i]; j < csr_row_ptr_D[i + 1]; j++)
                {
                    int col_D = csr_col_ind_D[j];
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        csr_row_ptr_C[i + 1]++;
                    }
                }
            }
        }
        for(int i = 0; i < m; i++)
            csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
        *nnz_C = csr_row_ptr_C[m];
    }

    template <typename T>
    static void host_csrgemm_impl(int        m,
                                  int        n,
                                  int        k,
                                  int        nnz_A,
                                  int        nnz_B,
                                  int        nnz_D,
                                  T          alpha,
                                  const int* csr_row_ptr_A,
                                  const int* csr_col_ind_A,
                                  const T*   csr_val_A,
                                  const int* csr_row_ptr_B,
                                  const int* csr_col_ind_B,
                                  const T*   csr_val_B,
                                  T          beta,
                                  const int* csr_row_ptr_D,
                                  const int* csr_col_ind_D,
                                  const T*   csr_val_D,
                                  const int* csr_row_ptr_C,
                                  int*       csr_col_ind_C,
                                  T*         csr_val_C)
    {
        ROUTINE_TRACE("host_csrgemm_impl");
        std::vector<int> nnzs(n, -1);
        for(int i = 0; i < m; i++)
        {
            int row_begin_C = csr_row_ptr_C[i];
            int row_end_C   = row_begin_C;
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                int col_A = csr_col_ind_A[j];
                T   val_A = alpha * csr_val_A[j];
                for(int p = csr_row_ptr_B[col_A]; p < csr_row_ptr_B[col_A + 1]; p++)
                {
                    int col_B = csr_col_ind_B[p];
                    T   val_B = csr_val_B[p];
                    if(nnzs[col_B] < row_begin_C)
                    {
                        nnzs[col_B]              = row_end_C;
                        csr_col_ind_C[row_end_C] = col_B;
                        csr_val_C[row_end_C]     = val_A * val_B;
                        row_end_C++;
                    }
                    else
                    {
                        csr_val_C[nnzs[col_B]] += val_A * val_B;
                    }
                }
            }
            if(beta != 0.0)
            {
                for(int j = csr_row_ptr_D[i]; j < csr_row_ptr_D[i + 1]; j++)
                {
                    int col_D = csr_col_ind_D[j];
                    T   val_D = beta * csr_val_D[j];
                    if(nnzs[col_D] < row_begin_C)
                    {
                        nnzs[col_D]              = row_end_C;
                        csr_col_ind_C[row_end_C] = col_D;
                        csr_val_C[row_end_C]     = val_D;
                        row_end_C++;
                    }
                    else
                    {
                        csr_val_C[nnzs[col_D]] += val_D;
                    }
                }
            }
        }
        int              nnz = csr_row_ptr_C[m];
        std::vector<int> cols(nnz);
        std::vector<T>   vals(nnz);
        memcpy(cols.data(), csr_col_ind_C, sizeof(int) * nnz);
        memcpy(vals.data(), csr_val_C, sizeof(double) * nnz);
        for(int i = 0; i < m; i++)
        {
            int              row_begin = csr_row_ptr_C[i];
            int              row_nnz   = csr_row_ptr_C[i + 1] - row_begin;
            std::vector<int> perm(row_nnz);
            for(int j = 0; j < row_nnz; j++)
                perm[j] = j;
            int* col_entry = cols.data() + row_begin;
            T*   val_entry = vals.data() + row_begin;
            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                return col_entry[a] < col_entry[b];
            });
            for(int j = 0; j < row_nnz; j++)
            {
                csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
                csr_val_C[row_begin + j]     = val_entry[perm[j]];
            }
        }
    }
}

void linalg::host_csrgemm_nnz(const csr_matrix& A,
                              const csr_matrix& B,
                              const csr_matrix& D,
                              csr_matrix&       C,
                              csrgemm_algorithm alg,
                              csrgemm_descr*    descr)
{
    ROUTINE_TRACE("linalg::host_csrgemm_nnz");
    C.resize(A.get_m(), B.get_n(), 0);
    int nnz_C;
    host_csrgemm_nnz_impl(A.get_m(),
                          B.get_n(),
                          A.get_n(),
                          A.get_nnz(),
                          B.get_nnz(),
                          0,
                          1.0,
                          A.get_row_ptr(),
                          A.get_col_ind(),
                          B.get_row_ptr(),
                          B.get_col_ind(),
                          0.0,
                          nullptr,
                          nullptr,
                          C.get_row_ptr(),
                          &nnz_C);
    C.resize(A.get_m(), B.get_n(), nnz_C);
}

void linalg::host_csrgemm_solve(double               alpha,
                                const csr_matrix&    A,
                                const csr_matrix&    B,
                                double               beta,
                                const csr_matrix&    D,
                                csr_matrix&          C,
                                csrgemm_algorithm    alg,
                                const csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csrgemm_solve");
    host_csrgemm_impl(A.get_m(),
                      B.get_n(),
                      A.get_n(),
                      A.get_nnz(),
                      B.get_nnz(),
                      0,
                      alpha,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      B.get_row_ptr(),
                      B.get_col_ind(),
                      B.get_val(),
                      beta,
                      nullptr,
                      nullptr,
                      static_cast<double*>(nullptr),
                      C.get_row_ptr(),
                      C.get_col_ind(),
                      C.get_val());
}
