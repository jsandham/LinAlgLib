#include <vector>

#include "../../trace.h"
#include "host_csrilu0.h"

namespace linalg
{
    template <typename T>
    static T get_diagonal_value_local(
        int col, int diag_index, const T* csr_val, int* structural_zero, int* numeric_zero)
    {
        T diag_val = 1.0;
        if(diag_index == -1)
        {
            if(structural_zero != nullptr)
                *structural_zero = std::min(*structural_zero, col);
        }
        else
        {
            diag_val = csr_val[diag_index];
            if(diag_val == 0.0)
            {
                if(numeric_zero != nullptr)
                    *numeric_zero = std::min(*numeric_zero, col);
                diag_val = 1.0;
            }
        }
        return diag_val;
    }

    template <typename T>
    static void host_csrilu0_impl(int        m,
                                  int        n,
                                  int        nnz,
                                  const int* csr_row_ptr,
                                  const int* csr_col_ind,
                                  T*         csr_val,
                                  int*       structural_zero,
                                  int*       numeric_zero)
    {
        ROUTINE_TRACE("host_csrilu0_impl");
        std::vector<int> diag_ptr(m, -1);
        for(int row = 0; row < m; row++)
        {
            int              row_begin = csr_row_ptr[row];
            int              row_end   = csr_row_ptr[row + 1];
            std::vector<int> col_offset_map(n, -1);
            for(int j = row_begin; j < row_end; j++)
                col_offset_map[csr_col_ind[j]] = j;
            for(int j = row_begin; j < row_end; j++)
            {
                int col_j = csr_col_ind[j];
                if(col_j < row)
                {
                    int diag_index = diag_ptr[col_j];
                    T   diag_val   = get_diagonal_value_local(
                        col_j, diag_index, csr_val, structural_zero, numeric_zero);
                    int row_end_col_j = csr_row_ptr[col_j + 1];
                    csr_val[j]        = csr_val[j] / diag_val;
                    for(int k = diag_index + 1; k < row_end_col_j; k++)
                    {
                        int col_k       = csr_col_ind[k];
                        int col_k_index = col_offset_map[col_k];
                        if(col_k_index != -1)
                            csr_val[col_k_index] = csr_val[col_k_index] - csr_val[j] * csr_val[k];
                    }
                }
                else if(col_j == row)
                {
                    diag_ptr[row] = j;
                    break;
                }
                else
                {
                    break;
                }
            }
        }
    }
}

void linalg::host_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::host_csrilu0");
    host_csrilu0_impl(LU.get_m(),
                      LU.get_n(),
                      LU.get_nnz(),
                      LU.get_row_ptr(),
                      LU.get_col_ind(),
                      LU.get_val(),
                      structural_zero,
                      numeric_zero);
}

void linalg::host_csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csrilu0_analysis");
}

void linalg::host_csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csrilu0_compute");
    host_csrilu0_impl(A.get_m(),
                      A.get_n(),
                      A.get_nnz(),
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      nullptr,
                      nullptr);
}
