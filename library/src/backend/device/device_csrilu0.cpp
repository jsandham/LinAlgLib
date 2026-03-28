#include "device_csrilu0.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_csrilu0.h"
#endif

void linalg::device_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    std::cout << "Error: csrilu0 on device not implemented" << std::endl;
}

void linalg::allocate_csrilu0_device_data(csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrilu0_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrilu0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::free_csrilu0_device_data(csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrilu0_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrilu0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrilu0_analysis");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrilu0_analysis(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrilu0_compute");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrilu0_compute(A.get_m(),
                                       A.get_n(),
                                       A.get_nnz(),
                                       A.get_row_ptr(),
                                       A.get_col_ind(),
                                       A.get_val(),
                                       descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
