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

#include "device_csrgemm.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_csrgemm.h"
#endif

void linalg::free_csrgemm_device_data(csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrgemm_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrgemm_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_csrgemm_nnz(const csr_matrix& A,
                                const csr_matrix& B,
                                const csr_matrix& D,
                                csr_matrix&       C,
                                csrgemm_algorithm alg,
                                csrgemm_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrgemm_nnz");
    if constexpr(is_cuda_available())
    {
        int nnz_C;
        CALL_CUDA(cuda_csrgemm_nnz(C.get_m(),
                                   C.get_n(),
                                   B.get_m(),
                                   A.get_nnz(),
                                   B.get_nnz(),
                                   0,
                                   descr,
                                   1.0,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   B.get_row_ptr(),
                                   B.get_col_ind(),
                                   0.0,
                                   nullptr,
                                   nullptr,
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

void linalg::device_csrgemm_solve(double               alpha,
                                  const csr_matrix&    A,
                                  const csr_matrix&    B,
                                  double               beta,
                                  const csr_matrix&    D,
                                  csr_matrix&          C,
                                  csrgemm_algorithm    alg,
                                  const csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrgemm_solve");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrgemm_solve(C.get_m(),
                                     C.get_n(),
                                     B.get_m(),
                                     A.get_nnz(),
                                     B.get_nnz(),
                                     0,
                                     C.get_nnz(),
                                     descr,
                                     alpha,
                                     A.get_row_ptr(),
                                     A.get_col_ind(),
                                     A.get_val(),
                                     B.get_row_ptr(),
                                     B.get_col_ind(),
                                     B.get_val(),
                                     beta,
                                     nullptr,
                                     nullptr,
                                     nullptr,
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
