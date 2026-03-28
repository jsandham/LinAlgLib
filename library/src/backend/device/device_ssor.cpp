#include "device_ssor.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_ssor.h"
#endif

void linalg::device_ssor_fill_lower_precond(const csr_matrix& A, csr_matrix& L, double omega)
{
    ROUTINE_TRACE("linalg::device_ssor_fill_lower_precond");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_ssor_fill_lower_precond(A.get_m(),
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
                                               omega));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_ssor_fill_upper_precond(const csr_matrix& A, csr_matrix& U, double omega)
{
    ROUTINE_TRACE("linalg::device_ssor_fill_upper_precond");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_ssor_fill_upper_precond(A.get_m(),
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
                                               omega));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
