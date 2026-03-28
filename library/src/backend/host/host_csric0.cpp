#include <cmath>
#include <iostream>
#include <vector>

#include "../../trace.h"
#include "host_csric0.h"

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
    static void host_csric0_impl(int        m,
                                 int        n,
                                 int        nnz,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 T*         csr_val,
                                 int*       structural_zero,
                                 int*       numeric_zero)
    {
        ROUTINE_TRACE("host_csric0_impl");
        std::cout << "Warning: host_csric0_impl is not optimized for performance." << std::endl;
        std::vector<int> diag_ptr(m, -1);
        for(int row = 0; row < m; row++)
        {
            int              row_begin = csr_row_ptr[row];
            int              row_end   = csr_row_ptr[row + 1];
            std::vector<int> col_offset_map(n, -1);
            for(int j = row_begin; j < row_end; j++)
                col_offset_map[csr_col_ind[j]] = j;
            T sum = 0.0;
            for(int j = row_begin; j < row_end; j++)
            {
                int col_j = csr_col_ind[j];
                if(col_j < row)
                {
                    int diag_index = diag_ptr[col_j];
                    T   s          = 0.0;
                    for(int k = csr_row_ptr[col_j]; k < diag_index; k++)
                    {
                        int col_k       = csr_col_ind[k];
                        int col_k_index = col_offset_map[col_k];
                        if(col_k_index != -1)
                            s += csr_val[col_k_index] * csr_val[k];
                    }
                    T diag_val = get_diagonal_value_local(
                        col_j, diag_index, csr_val, structural_zero, numeric_zero);
                    T val = (csr_val[j] - s) / diag_val;
                    sum += val * val;
                    csr_val[j] = val;
                }
                else if(col_j == row)
                {
                    diag_ptr[row] = j;
                    csr_val[j]    = std::sqrt(std::abs(csr_val[j] - sum));
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

void linalg::host_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::host_csric0");
    host_csric0_impl(LL.get_m(),
                     LL.get_n(),
                     LL.get_nnz(),
                     LL.get_row_ptr(),
                     LL.get_col_ind(),
                     LL.get_val(),
                     structural_zero,
                     numeric_zero);
}

void linalg::host_csric0_analysis(const csr_matrix& A, csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csric0_analysis");
}

void linalg::host_csric0_compute(csr_matrix& A, const csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::host_csric0_compute");
    host_csric0_impl(A.get_m(),
                     A.get_n(),
                     A.get_nnz(),
                     A.get_row_ptr(),
                     A.get_col_ind(),
                     A.get_val(),
                     nullptr,
                     nullptr);
    A.print_matrix("A after IC factorization");
}
