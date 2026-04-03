//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************

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
