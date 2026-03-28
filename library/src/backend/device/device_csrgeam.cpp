#include "device_csrgeam.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_csrgeam.h"
#endif

void linalg::allocate_csrgeam_device_data(csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrgeam_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrgeam_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::free_csrgeam_device_data(csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrgeam_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrgeam_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrgeam_nnz(const csr_matrix& A,
                                const csr_matrix& B,
                                csr_matrix&       C,
                                csrgeam_algorithm alg,
                                csrgeam_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrgeam_nnz");
    if constexpr(is_cuda_available())
    {
        int nnz_C;
        CALL_CUDA(cuda_csrgeam_nnz(C.get_m(),
                                   C.get_n(),
                                   A.get_nnz(),
                                   B.get_nnz(),
                                   descr,
                                   1.0,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   0.0,
                                   B.get_row_ptr(),
                                   B.get_col_ind(),
                                   C.get_row_ptr(),
                                   &nnz_C));
        C.resize(C.get_m(), C.get_n(), nnz_C);
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrgeam_solve(double               alpha,
                                  const csr_matrix&    A,
                                  double               beta,
                                  const csr_matrix&    B,
                                  csr_matrix&          C,
                                  csrgeam_algorithm    alg,
                                  const csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrgeam_solve");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrgeam_solve(C.get_m(),
                                     C.get_n(),
                                     A.get_nnz(),
                                     B.get_nnz(),
                                     C.get_nnz(),
                                     descr,
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
                                     C.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
