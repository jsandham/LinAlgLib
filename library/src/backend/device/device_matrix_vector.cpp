#include "device_matrix_vector.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_matrix_vector.h"
#endif

void linalg::device_compute_residual(const csr_matrix&     A,
                                     const vector<double>& x,
                                     const vector<double>& b,
                                     vector<double>&       res)
{
    ROUTINE_TRACE("linalg::device_compute_residual");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_compute_residual(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        x.get_vec(),
                                        b.get_vec(),
                                        res.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::allocate_csrmv_device_data(csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrmv_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrmv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::free_csrmv_device_data(csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrmv_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrmv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrmv_analysis");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv_analysis(A.get_m(),
                                      A.get_n(),
                                      A.get_nnz(),
                                      A.get_row_ptr(),
                                      A.get_col_ind(),
                                      A.get_val(),
                                      alg,
                                      descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrmv_solve(double                alpha,
                                const csr_matrix&     A,
                                const vector<double>& x,
                                double                beta,
                                vector<double>&       y,
                                csrmv_algorithm       alg,
                                const csrmv_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrmv_solve");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv_solve(A.get_m(),
                                   A.get_n(),
                                   A.get_nnz(),
                                   alpha,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   A.get_val(),
                                   x.get_vec(),
                                   beta,
                                   y.get_vec(),
                                   alg,
                                   descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
