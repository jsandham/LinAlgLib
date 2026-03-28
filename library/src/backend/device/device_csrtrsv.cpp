#include "device_csrtrsv.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_csrtrsv.h"
#endif

void linalg::allocate_csrtrsv_device_data(csrtrsv_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrtrsv_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrtrsv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::free_csrtrsv_device_data(csrtrsv_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrtrsv_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrtrsv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrtrsv_analysis(const csr_matrix& A,
                                     triangular_type   tri_type,
                                     diagonal_type     diag_type,
                                     csrtrsv_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrtrsv_analysis");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrtrsv_analysis(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        tri_type,
                                        diag_type,
                                        descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrtrsv_solve(const csr_matrix&     A,
                                  const vector<double>& b,
                                  vector<double>&       x,
                                  double                alpha,
                                  triangular_type       tri_type,
                                  diagonal_type         diag_type,
                                  const csrtrsv_descr*  descr)
{
    ROUTINE_TRACE("linalg::device_csrtrsv_solve");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrtrsv_solve(A.get_m(),
                                     A.get_n(),
                                     A.get_nnz(),
                                     alpha,
                                     A.get_row_ptr(),
                                     A.get_col_ind(),
                                     A.get_val(),
                                     b.get_vec(),
                                     x.get_vec(),
                                     tri_type,
                                     diag_type,
                                     descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
