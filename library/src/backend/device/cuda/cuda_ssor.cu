//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

#include "cuda_ssor.h"
#include "cuda_extract.h"

#include "preconditioner_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// SSOR fill lower preconditioner L = (beta * D - E) * D^-1, beta = 1 / omega
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_ssor_fill_lower_precond(int        m_A,
                                          int        n_A,
                                          int        nnz_A,
                                          const int* csr_row_ptr_A,
                                          const int* csr_col_ind_A,
                                          const T*   csr_val_A,
                                          int        m_L,
                                          int        n_L,
                                          int        nnz_L,
                                          const int* csr_row_ptr_L,
                                          int*       csr_col_ind_L,
                                          T*         csr_val_L,
                                          T          omega)
{
    ROUTINE_TRACE("linalg::cuda_ssor_fill_lower_precond");

    T* diag = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&diag, sizeof(T) * m_A));

    cuda_extract_diagonal(m_A, n_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_val_A, diag);

    ssor_fill_lower_precond_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                        n_A,
                                                                        nnz_A,
                                                                        omega,
                                                                        csr_row_ptr_A,
                                                                        csr_col_ind_A,
                                                                        csr_val_A,
                                                                        diag,
                                                                        csr_row_ptr_L,
                                                                        csr_col_ind_L,
                                                                        csr_val_L);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaFree(diag));
}

//-------------------------------------------------------------------------------
// SSOR fill upper preconditioner U = (beta * D - F), beta = 1 / omega
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_ssor_fill_upper_precond(int        m_A,
                                          int        n_A,
                                          int        nnz_A,
                                          const int* csr_row_ptr_A,
                                          const int* csr_col_ind_A,
                                          const T*   csr_val_A,
                                          int        m_U,
                                          int        n_U,
                                          int        nnz_U,
                                          const int* csr_row_ptr_U,
                                          int*       csr_col_ind_U,
                                          T*         csr_val_U,
                                          T          omega)
{
    ROUTINE_TRACE("linalg::cuda_ssor_fill_upper_precond");
    ssor_fill_upper_precond_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                        n_A,
                                                                        nnz_A,
                                                                        omega,
                                                                        csr_row_ptr_A,
                                                                        csr_col_ind_A,
                                                                        csr_val_A,
                                                                        csr_row_ptr_U,
                                                                        csr_col_ind_U,
                                                                        csr_val_U);
    CHECK_CUDA_LAUNCH_ERROR();
}

// Explicit instantiations
template void linalg::cuda_ssor_fill_lower_precond<double>(int, int, int, const int*, const int*, const double*, int, int, int, const int*, int*, double*, double);
template void linalg::cuda_ssor_fill_lower_precond<float>(int, int, int, const int*, const int*, const float*, int, int, int, const int*, int*, float*, float);
template void linalg::cuda_ssor_fill_upper_precond<double>(int, int, int, const int*, const int*, const double*, int, int, int, const int*, int*, double*, double);
template void linalg::cuda_ssor_fill_upper_precond<float>(int, int, int, const int*, const int*, const float*, int, int, int, const int*, int*, float*, float);
