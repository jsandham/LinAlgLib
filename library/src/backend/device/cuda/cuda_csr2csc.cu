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

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cuda_csr2csc.h"

#include "csr2coo_kernels.cuh"

void linalg::cuda_csr2csc_buffer_size(int           m,
                                      int           n,
                                      int           nnz,
                                      const int*    csr_row_ptr,
                                      const int*    csr_Col_ind,
                                      const double* csr_val,
                                      size_t*       buffer_size)
{
    *buffer_size = 0;
    *buffer_size += sizeof(int) * nnz; // perm
    *buffer_size += sizeof(int) * nnz; // coo_row_ind
}

void linalg::cuda_csr2csc(int           m,
                          int           n,
                          int           nnz,
                          const int*    csr_row_ptr,
                          const int*    csr_col_ind,
                          const double* csr_val,
                          int*          csc_col_ptr,
                          int*          csc_row_ind,
                          double*       csc_val,
                          void*         buffer)
{
    int* perm        = reinterpret_cast<int*>(buffer);
    int* coo_row_ind = reinterpret_cast<int*>(buffer) + nnz;

    fill_identity_permuation_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(nnz, perm);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(csc_row_ind, csr_col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice));

    // Sort keys and apply the same permutation to values directly on device pointers.
    thrust::sort_by_key(thrust::device, csc_row_ind, csc_row_ind + nnz, perm);

    coo2csr_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(m, n, nnz, csc_row_ind, csc_col_ptr);
    CHECK_CUDA_LAUNCH_ERROR();

    csr2coo_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(m, n, nnz, csr_row_ptr, coo_row_ind);
    CHECK_CUDA_LAUNCH_ERROR();

    csr2csc_permute_colval_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(
        m, n, nnz, coo_row_ind, csr_val, perm, csc_row_ind, csc_val);
    CHECK_CUDA_LAUNCH_ERROR();
}
