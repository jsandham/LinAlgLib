#include "host_csr2csc.h"
#include "../../trace.h"

void linalg::host_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::host_transpose_matrix");
    transposeA.resize(A.get_n(), A.get_m(), A.get_nnz());
    int*    csr_row_ptr_T = transposeA.get_row_ptr();
    int*    csr_col_ind_T = transposeA.get_col_ind();
    double* csr_val_T     = transposeA.get_val();
    for(size_t i = 0; i < transposeA.get_m() + 1; i++)
    {
        csr_row_ptr_T[i] = 0;
    }
    for(size_t i = 0; i < transposeA.get_nnz(); i++)
    {
        csr_col_ind_T[i] = -1;
    }
    const int*    csr_row_ptr_A = A.get_row_ptr();
    const int*    csr_col_ind_A = A.get_col_ind();
    const double* csr_val_A     = A.get_val();
    for(int i = 0; i < A.get_m(); i++)
    {
        for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
        {
            csr_row_ptr_T[csr_col_ind_A[j] + 1]++;
        }
    }
    for(int i = 0; i < transposeA.get_m(); i++)
    {
        csr_row_ptr_T[i + 1] += csr_row_ptr_T[i];
    }
    for(int i = 0; i < A.get_m(); i++)
    {
        for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
        {
            int col = csr_col_ind_A[j];
            for(int k = csr_row_ptr_T[col]; k < csr_row_ptr_T[col + 1]; k++)
            {
                if(csr_col_ind_T[k] == -1)
                {
                    csr_col_ind_T[k] = i;
                    csr_val_T[k]     = csr_val_A[j];
                    break;
                }
            }
        }
    }
}
