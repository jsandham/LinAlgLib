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

#include <cassert>
#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "cuda_csrtrsv.h"

#include "csr2coo_kernels.cuh"
#include "csrtrsv_kernels.cuh"

struct linalg::csrtrsv_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::allocate_csrtrsv_cuda_data(csrtrsv_descr* descr)
{
    descr->done_array = nullptr;
    descr->row_perm   = nullptr;
    descr->diag_ind   = nullptr;
}

void linalg::free_csrtrsv_cuda_data(csrtrsv_descr* descr)
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

void linalg::cuda_csrtrsv_analysis(int             m,
                                   int             n,
                                   int             nnz,
                                   const int*      csr_row_ptr,
                                   const int*      csr_col_ind,
                                   const double*   csr_val,
                                   triangular_type tri_type,
                                   diagonal_type   diag_type,
                                   csrtrsv_descr*  descr)
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

    csrtrsv_analysis_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
        m, tri_type, csr_row_ptr, csr_col_ind, csr_val, descr->diag_ind, descr->done_array);
    CHECK_CUDA_LAUNCH_ERROR();

    // std::vector<int> hdiag_ind(m, 10);
    // CHECK_CUDA(
    //     cudaMemcpy(hdiag_ind.data(), descr->diag_ind, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "diag_ind" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdiag_ind[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // std::vector<int> hdone_array(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hdone_array.data(), descr->done_array, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "done_array" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdone_array[i] << " ";
    // }
    // std::cout << "" << std::endl;

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(descr->done_array);
    thrust::device_ptr<int> d_values(descr->row_perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    // std::vector<int> hrow_perm(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hrow_perm.data(), descr->row_perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "row_perm" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hrow_perm[i] << " ";
    // }
    // std::cout << "" << std::endl;
}
void linalg::cuda_csrtrsv_solve(int                  m,
                                int                  n,
                                int                  nnz,
                                double               alpha,
                                const int*           csr_row_ptr,
                                const int*           csr_col_ind,
                                const double*        csr_val,
                                const double*        b,
                                double*              x,
                                triangular_type      tri_type,
                                diagonal_type        diag_type,
                                const csrtrsv_descr* descr)
{
    assert(descr->diag_ind != nullptr);
    assert(descr->done_array != nullptr);
    assert(descr->row_perm != nullptr);

    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    csrtrsv_solve_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m,
                                                                       tri_type,
                                                                       diag_type,
                                                                       alpha,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       csr_val,
                                                                       descr->diag_ind,
                                                                       b,
                                                                       x,
                                                                       descr->done_array,
                                                                       descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // std::vector<double> hx(m, 0.0);
    // CHECK_CUDA(cudaMemcpy(hx.data(), x, sizeof(double) * m, cudaMemcpyDeviceToHost));
    // std::cout << "x" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hx[i] << " ";
    // }
    // std::cout << std::endl;
}
