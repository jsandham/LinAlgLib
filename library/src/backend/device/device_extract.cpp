#include "device_extract.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_extract.h"
#endif

void linalg::device_diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::device_diagonal");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_diagonal(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        d.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L)
{
    ROUTINE_TRACE("linalg::device_extract_lower_triangular_nnz");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_lower_triangular_nnz(A.get_m(),
                                                    A.get_n(),
                                                    A.get_nnz(),
                                                    A.get_row_ptr(),
                                                    A.get_col_ind(),
                                                    L.get_row_ptr(),
                                                    &nnz_L));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_extract_lower_triangular(const csr_matrix& A, csr_matrix& L)
{
    ROUTINE_TRACE("linalg::device_extract_lower_triangular");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_lower_triangular(A.get_m(),
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
                                                L.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U)
{
    ROUTINE_TRACE("linalg::device_extract_upper_triangular_nnz_count");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_upper_triangular_nnz(A.get_m(),
                                                    A.get_n(),
                                                    A.get_nnz(),
                                                    A.get_row_ptr(),
                                                    A.get_col_ind(),
                                                    U.get_row_ptr(),
                                                    &nnz_U));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_extract_upper_triangular(const csr_matrix& A, csr_matrix& U)
{
    ROUTINE_TRACE("linalg::device_extract_upper_triangular");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_upper_triangular(A.get_m(),
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
                                                U.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
