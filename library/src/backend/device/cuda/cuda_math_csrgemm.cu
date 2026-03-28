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

#include <iostream>
#include <vector>

#include <thrust/sort.h>

#include "cuda_math.h"
#include "cuda_primitives.h"

#include "csr2coo_kernels.cuh"
#include "csrgemm_kernels.cuh"

struct linalg::csrgemm_descr
{
    int* perm;
    int* bin_offsets;
};

void linalg::allocate_csrgemm_cuda_data(csrgemm_descr* descr)
{
    descr->perm        = nullptr;
    descr->bin_offsets = nullptr;
}

void linalg::free_csrgemm_cuda_data(csrgemm_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->perm != nullptr)
        {
            std::cout << "Freeing perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->perm));
        }

        if(descr->bin_offsets != nullptr)
        {
            std::cout << "Freeing bin_offsets" << std::endl;
            CHECK_CUDA(cudaFree(descr->bin_offsets));
        }
    }
}

void linalg::cuda_csrgemm_nnz(int            m,
                              int            n,
                              int            k,
                              int            nnz_A,
                              int            nnz_B,
                              int            nnz_D,
                              csrgemm_descr* descr,
                              double         alpha,
                              const int*     csr_row_ptr_A,
                              const int*     csr_col_ind_A,
                              const int*     csr_row_ptr_B,
                              const int*     csr_col_ind_B,
                              double         beta,
                              const int*     csr_row_ptr_D,
                              const int*     csr_col_ind_D,
                              int*           csr_row_ptr_C,
                              int*           nnz_C)
{
    // Determine maximum hash table size
    csrgemm_count_products_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
        m, alpha, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_B, beta, csr_row_ptr_D, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hcsr_row_ptr_C(m + 1);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    descr->perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->perm, sizeof(int) * m));

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(csr_row_ptr_C);
    thrust::device_ptr<int> d_values(descr->perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    std::vector<int> hperm(m);
    CHECK_CUDA(cudaMemcpy(hperm.data(), descr->perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "perm" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hperm[i] << " ";
    }
    std::cout << "" << std::endl;

    // Bin rows based on required hash table size.
    // Bin sizes are: 32, 64, 128, 256, 512, 1024, 2048, 4096
    int bin_count      = 8;
    descr->bin_offsets = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->bin_offsets, sizeof(int) * (bin_count + 1)));
    CHECK_CUDA(cudaMemset(descr->bin_offsets, 0, sizeof(int) * (bin_count + 1)));

    compute_rows_bin_number_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    fill_bin_offsets_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C, descr->bin_offsets);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    CHECK_CUDA(cudaMemset(csr_row_ptr_C, 0, sizeof(int) * (m + 1)));

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 1024>
            <<<((bin_5_count - 1) / (256 / 32) + 1), 256>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgemm_nnz_per_row_kernel<128, 32, 2048>
            <<<((bin_6_count - 1) / (128 / 32) + 1), 128>>>(bin_6_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[6]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgemm_nnz_per_row_kernel<64, 32, 4096>
            <<<((bin_7_count - 1) / (64 / 32) + 1), 64>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_row_ptr_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    cuda_exclusive_scan(m + 1, csr_row_ptr_C);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    CHECK_CUDA(cudaMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(int), cudaMemcpyDeviceToHost));

    // CHECK_CUDA(cudaFree(perm));
    // CHECK_CUDA(cudaFree(bin_offsets));
}

void linalg::cuda_csrgemm_solve(int                  m,
                                int                  n,
                                int                  k,
                                int                  nnz_A,
                                int                  nnz_B,
                                int                  nnz_D,
                                int                  nnz_C,
                                const csrgemm_descr* descr,
                                double               alpha,
                                const int*           csr_row_ptr_A,
                                const int*           csr_col_ind_A,
                                const double*        csr_val_A,
                                const int*           csr_row_ptr_B,
                                const int*           csr_col_ind_B,
                                const double*        csr_val_B,
                                double               beta,
                                const int*           csr_row_ptr_D,
                                const int*           csr_col_ind_D,
                                const double*        csr_val_D,
                                const int*           csr_row_ptr_C,
                                int*                 csr_col_ind_C,
                                double*              csr_val_C)
{
    std::cout << "nnz_C: " << nnz_C << std::endl;

    int              bin_count = 8;
    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgemm_fill_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgemm_fill_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgemm_fill_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgemm_fill_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgemm_fill_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgemm_fill_kernel<128, 32, 1024>
            <<<((bin_5_count - 1) / (128 / 32) + 1), 128>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgemm_fill_kernel<64, 32, 2048>
            <<<((bin_6_count - 1) / (64 / 32) + 1), 64>>>(bin_6_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[6]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgemm_fill_kernel<32, 32, 4096>
            <<<((bin_7_count - 1) / (32 / 32) + 1), 32>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int>    hcsr_row_ptr_C(m + 1, 0);
    std::vector<int>    hcsr_col_ind_C(nnz_C, 0);
    std::vector<double> hcsr_val_C(nnz_C);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        hcsr_col_ind_C.data(), csr_col_ind_C, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(hcsr_val_C.data(), csr_val_C, sizeof(double) * nnz_C, cudaMemcpyDeviceToHost));

    std::cout << "hcsr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_col_ind_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_col_ind_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_val_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_val_C[i] << " ";
    }
    std::cout << "" << std::endl;
}
