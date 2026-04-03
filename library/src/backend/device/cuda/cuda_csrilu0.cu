//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#include <cassert>
#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "cuda_csrilu0.h"

#include "csr2coo_kernels.cuh"
#include "csrtrsv_kernels.cuh"

//-------------------------------------------------------------------------------
// Compute incomplete LU factorization inplace
//-------------------------------------------------------------------------------
void linalg::cuda_csrilu0(int        m,
                          int        n,
                          int        nnz,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          double*    csr_val,
                          int*       structural_zero,
                          int*       numeric_zero)
{
}

//-------------------------------------------------------------------------------
// Compute Incomplete LU ILU0: A = L * U
//-------------------------------------------------------------------------------
struct linalg::csrilu0_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::free_csrilu0_cuda_data(csrilu0_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->done_array != nullptr)
        {
            std::cout << "Freeing done_array" << std::endl;
            CHECK_CUDA(cudaFree(descr->done_array));
        }

        if(descr->row_perm != nullptr)
        {
            std::cout << "Freeing row_perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->row_perm));
        }

        if(descr->diag_ind != nullptr)
        {
            std::cout << "Freeing diag_ind" << std::endl;
            CHECK_CUDA(cudaFree(descr->diag_ind));
        }
    }
}

void linalg::cuda_csrilu0_analysis(int            m,
                                   int            n,
                                   int            nnz,
                                   const int*     csr_row_ptr,
                                   const int*     csr_col_ind,
                                   const double*  csr_val,
                                   csrilu0_descr* descr)
{
    std::cout << "csrtrsv_analysis m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

    // Free any previous allocations?
    assert(descr->done_array == nullptr);
    assert(descr->row_perm == nullptr);
    assert(descr->diag_ind == nullptr);

    descr->done_array = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->done_array), sizeof(int) * m));
    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    descr->row_perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->row_perm), sizeof(int) * m));

    descr->diag_ind = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->diag_ind), sizeof(int) * m));

    csrtrsv_analysis_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m,
                                                                          triangular_type::lower,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          csr_val,
                                                                          descr->diag_ind,
                                                                          descr->done_array);
    CHECK_CUDA_LAUNCH_ERROR();

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(descr->done_array);
    thrust::device_ptr<int> d_values(descr->row_perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);
}

void linalg::cuda_csrilu0_compute(int                  m,
                                  int                  n,
                                  int                  nnz,
                                  const int*           csr_row_ptr,
                                  const int*           csr_col_ind,
                                  double*              csr_val,
                                  const csrilu0_descr* descr)
{
    assert(descr->diag_ind != nullptr);
    assert(descr->done_array != nullptr);
    assert(descr->row_perm != nullptr);

    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));
}
