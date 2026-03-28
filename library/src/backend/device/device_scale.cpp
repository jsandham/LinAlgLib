#include "device_scale.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_scale.h"
#endif

void linalg::device_scale_diagonal(csr_matrix& A, double scalar)
{
    ROUTINE_TRACE("linalg::device_scale_diagonal");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_scale_diagonal(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), scalar));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag)
{
    ROUTINE_TRACE("linalg::device_scale_by_inverse_diagonal");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_scale_by_inverse_diagonal(
            A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), diag.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
