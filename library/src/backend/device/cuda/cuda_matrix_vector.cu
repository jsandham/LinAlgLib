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

#include <array>
#include <assert.h>
#include <string>
#include <vector>

#include "cuda_matrix_vector.h"

#include "../../../descriptors/csrmv_descr_internal.h"

#include "compute_residual_kernels.cuh"
#include "csrmv_kernels.cuh"

#include "../../../trace.h"

namespace linalg
{
    static std::string csrmv_alg_to_string(csrmv_algorithm alg)
    {
        switch(alg)
        {
        case csrmv_algorithm::default_algorithm:
            return "default_algorithm";
        case csrmv_algorithm::rowsplit:
            return "rowsplit";
        case csrmv_algorithm::nnzsplit:
            return "nnzsplit";
        case csrmv_algorithm::merge_path:
            return "marge_path";
        case csrmv_algorithm::lrb:
            return "lrb";
        }

        return "invalid";
    }

    static void csrmv_analysis_lrb_dispatch(
        int m, int n, int nnz, const int* csr_row_ptr, const int* csr_col_ind, csrmv_descr* descr)
    {
        // Free any previous allocations?
        assert(descr->bin_start_ptr == nullptr);
        assert(descr->row_index_in_bin == nullptr);
        assert(descr->row_index_in_bin_sorted == nullptr);

        CHECK_CUDA(cudaMalloc((void**)&(descr->bin_count), sizeof(int) * 32));
        CHECK_CUDA(cudaMalloc((void**)&(descr->bin_start_ptr), sizeof(int) * (32 + 1)));
        CHECK_CUDA(cudaMalloc((void**)&(descr->row_index_in_bin), sizeof(int) * m));
        CHECK_CUDA(cudaMalloc((void**)&(descr->row_index_in_bin_sorted), sizeof(int) * m));

        CHECK_CUDA(cudaMemset(descr->bin_count, 0, sizeof(int) * 32));

        compute_analysis_pass1<256><<<((m - 1) / 256 + 1), 256>>>(
                                        m,
                                       csr_row_ptr,
                                       descr->bin_count,
                                       descr->row_index_in_bin);

        CHECK_CUDA(cudaMemcpy(descr->hbin_count.data(),
                              descr->bin_count,
                              sizeof(int) * 32,
                              cudaMemcpyDeviceToHost));

        compute_analysis_pass2<256><<<((m - 1) / 256 + 1), 256>>>(
                                        m,
                                       csr_row_ptr,
                                       descr->bin_count,
                                       descr->row_index_in_bin,
                                       descr->bin_start_ptr,
                                       descr->row_index_in_bin_sorted);








        // // Free any previous allocations?
        // assert(descr->bin_start_ptr == nullptr);
        // assert(descr->row_index_in_bin == nullptr);
        // assert(descr->row_index_in_bin_sorted == nullptr);

        // CHECK_CUDA(cudaMalloc((void**)&(descr->bin_start_ptr), sizeof(int) * (32 + 1)));
        // CHECK_CUDA(cudaMalloc((void**)&(descr->row_index_in_bin), sizeof(int) * m));
        // CHECK_CUDA(cudaMalloc((void**)&(descr->row_index_in_bin_sorted), sizeof(int) * m));

        // CHECK_CUDA(cudaMemset(descr->row_index_in_bin_sorted, 0, sizeof(int) * m));

        // for(int i = 0; i < 32; i++)
        // {
        //     descr->hbin_count[i] = 0;
        // }

        // std::vector<int> hcsr_row_ptr(m + 1);
        // CHECK_CUDA(cudaMemcpy(
        //     hcsr_row_ptr.data(), csr_row_ptr, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

        // std::vector<int>        row_index_in_bin(m, 0);
        // std::array<int, 32 + 1> bin_start_ptr = {};

        // for(int i = 0; i < m; i++)
        // {
        //     const int row_length = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
        //     const int bin        = (row_length != 0) ? std::ceil(std::log2(row_length)) : 0;

        //     row_index_in_bin[i] = descr->hbin_count[bin];
        //     descr->hbin_count[bin]++;
        //     bin_start_ptr[bin]++;
        // }

        // // std::cout << "descr->hbin_count" << std::endl;
        // // for(int i = 0; i < 32; i++)
        // // {
        // //     std::cout << descr->hbin_count[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // // std::cout << "bin_start_ptr" << std::endl;
        // // for(int i = 0; i < 32 + 1; i++)
        // // {
        // //     std::cout << bin_start_ptr[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // // std::cout << "row_index_in_bin" << std::endl;
        // // for(int i = 0; i < m; i++)
        // // {
        // //     std::cout << row_index_in_bin[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // int count = 0;
        // for(int i = 0; i < 32; i++)
        // {
        //     const int tmp    = bin_start_ptr[i];
        //     bin_start_ptr[i] = count;
        //     count += tmp;
        // }

        // // std::cout << "bin_start_ptr" << std::endl;
        // // for(int i = 0; i < 32 + 1; i++)
        // // {
        // //     std::cout << bin_start_ptr[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // std::vector<int> row_index_in_bin_sorted(m, 0);
        // for(int i = 0; i < m; i++)
        // {
        //     const int row_length = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
        //     const int bin        = (row_length != 0) ? std::ceil(std::log2(row_length)) : 0;

        //     row_index_in_bin_sorted[bin_start_ptr[bin] + row_index_in_bin[i]] = i;
        // }

        // // std::cout << "bin_start_ptr" << std::endl;
        // // for(int i = 0; i < 32 + 1; i++)
        // // {
        // //     std::cout << bin_start_ptr[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // // std::cout << "row_index_in_bin_sorted" << std::endl;
        // // for(int i = 0; i < m; i++)
        // // {
        // //     std::cout << row_index_in_bin_sorted[i] << " ";
        // // }
        // // std::cout << "" << std::endl;

        // CHECK_CUDA(cudaMemcpy(descr->bin_start_ptr,
        //                       bin_start_ptr.data(),
        //                       sizeof(int) * (32 + 1),
        //                       cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(descr->row_index_in_bin,
        //                       row_index_in_bin.data(),
        //                       sizeof(int) * m,
        //                       cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(descr->row_index_in_bin_sorted,
        //                       row_index_in_bin_sorted.data(),
        //                       sizeof(int) * m,
        //                       cudaMemcpyHostToDevice));
    }

    static void csrmv_analysis_merge_path_dispatch(
        int m, int n, int nnz, const int* csr_row_ptr, const int* csr_col_ind, csrmv_descr* descr)
    {
    }

    static void csrmv_analysis_algorithm_dispatch(int             m,
                                                  int             n,
                                                  int             nnz,
                                                  const int*      csr_row_ptr,
                                                  const int*      csr_col_ind,
                                                  csrmv_algorithm alg,
                                                  csrmv_descr*    descr)
    {
        switch(alg)
        {
        case csrmv_algorithm::default_algorithm:
        case csrmv_algorithm::rowsplit:
        case csrmv_algorithm::nnzsplit:
            break;
        case csrmv_algorithm::merge_path:
            csrmv_analysis_merge_path_dispatch(m, n, nnz, csr_row_ptr, csr_col_ind, descr);
            break;
        case csrmv_algorithm::lrb:
            csrmv_analysis_lrb_dispatch(m, n, nnz, csr_row_ptr, csr_col_ind, descr);
            break;
        default:
            throw std::runtime_error("Unknown csrmv_algorithm");
        }
    }

    template <typename T>
    static void csrmv_row_split_dispatch(int                m,
                                         int                n,
                                         int                nnz,
                                         T                  alpha,
                                         const int*         csr_row_ptr,
                                         const int*         csr_col_ind,
                                         const T*           csr_val,
                                         const T*           x,
                                         T                  beta,
                                         T*                 y,
                                         const csrmv_descr* descr)
    {
        const int avg_nnz_per_row = nnz / m;

        if(avg_nnz_per_row <= 8)
        {
            csrmv_row_split_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
        }
        else if(avg_nnz_per_row <= 16)
        {
            csrmv_row_split_kernel<256, 8><<<((m - 1) / (256 / 8) + 1), 256>>>(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
        }
        else if(avg_nnz_per_row <= 32)
        {
            csrmv_row_split_kernel<256, 16><<<((m - 1) / (256 / 16) + 1), 256>>>(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
        }
        else
        {
            csrmv_row_split_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
        }
    }

    template <typename T>
    static void csrmv_nnz_split_dispatch(int                m,
                                         int                n,
                                         int                nnz,
                                         T                  alpha,
                                         const int*         csr_row_ptr,
                                         const int*         csr_col_ind,
                                         const T*           csr_val,
                                         const T*           x,
                                         T                  beta,
                                         T*                 y,
                                         const csrmv_descr* descr)
    {
        CHECK_CUDA(cudaMemset(y, 0, sizeof(T) * m)); // need to call kernel to handle beta

        csrmv_nnz_split_kernel<256, 32, 8><<<((nnz - 1) / (8 * 256) + 1), 256>>>(
            m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    }

    template <typename T>
    static void csrmv_lrb_dispatch(int                m,
                                   int                n,
                                   int                nnz,
                                   T                  alpha,
                                   const int*         csr_row_ptr,
                                   const int*         csr_col_ind,
                                   const T*           csr_val,
                                   const T*           x,
                                   T                  beta,
                                   T*                 y,
                                   const csrmv_descr* descr)
    {
        // Short rows (bin sizes 2-16)
        for(int bin = 0; bin < 5; bin++)
        {
            if(descr->hbin_count[bin] > 0)
            {
                // std::cout << "small bin: " << bin << " hbin_count[bin]: " << descr->hbin_count[bin]
                //           << std::endl;
                csrmv_lrb_small_kernel<256>
                    <<<((descr->hbin_count[bin] - 1) / 256 + 1), 256>>>(m,
                                                   n,
                                                   nnz,
                                                   bin,
                                                   descr->hbin_count[bin],
                                                   alpha,
                                                   descr->bin_start_ptr,
                                                   descr->row_index_in_bin_sorted,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   csr_val,
                                                   x,
                                                   beta,
                                                   y);
            }
        }

        // Medium rows (one warp per row, bin sizes 32-128)
        for(int bin = 5; bin < 8; bin++)
        {
            if(descr->hbin_count[bin] > 0)
            {
                // int bin_size = 1 << bin;
                // std::cout << "medium bin: " << bin << " hbin_count[bin]: " << descr->hbin_count[bin]
                //           << " bin_size: " << bin_size << std::endl;
                csrmv_lrb_medium_kernel<256, 32>
                    <<<((descr->hbin_count[bin] - 1) / (256 / 32) + 1), 256>>>(
                        m,
                        n,
                        nnz,
                        bin,
                        descr->hbin_count[bin],
                        alpha,
                        descr->bin_start_ptr,
                        descr->row_index_in_bin_sorted,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        beta,
                        y);
            }
        }

        // Medium-large rows (one block per row, bin sizes 256-2^32)
        for(int bin = 8; bin < 32 /*14*/; bin++)
        {
            if(descr->hbin_count[bin] > 0)
            {
                // std::cout << "medium-large bin: " << bin
                //           << " hbin_count[bin]: " << descr->hbin_count[bin] << std::endl;
                csrmv_lrb_medium_large_kernel<256>
                    <<<((descr->hbin_count[bin] - 1) / 256 + 1), 256>>>(
                        m,
                        n,
                        nnz,
                        bin,
                        descr->hbin_count[bin],
                        alpha,
                        descr->bin_start_ptr,
                        descr->row_index_in_bin_sorted,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        beta,
                        y);
            }
        }

        // for(int bin = 0/*14*/; bin < 32; bin++)
        // {
        //     if(descr->hbin_count[bin] > 0)
        //     {
        //         // How many blocks do I need?
        //         const int bin_size = 1 << bin;

        //         const int num_blocks = descr->hbin_count[bin] * (bin_size / 256);

        //         csrmv_lrb_large_kernel<256>
        //             <<<num_blocks, 256>>>(
        //                 m,
        //                 n,
        //                 nnz,
        //                 bin,
        //                 descr->hbin_count[bin],
        //                 alpha,
        //                 descr->bin_start_ptr,
        //                 descr->row_index_in_bin_sorted,
        //                 csr_row_ptr,
        //                 csr_col_ind,
        //                 csr_val,
        //                 x,
        //                 beta,
        //                 y);
        //     }
        // }
    }

    template <typename T>
    static void csrmv_merge_path_dispatch(int                m,
                                          int                n,
                                          int                nnz,
                                          T                  alpha,
                                          const int*         csr_row_ptr,
                                          const int*         csr_col_ind,
                                          const T*           csr_val,
                                          const T*           x,
                                          T                  beta,
                                          T*                 y,
                                          const csrmv_descr* descr)
    {
    }

    template <typename T>
    static void csrmv_algorithm_dispatch(int                m,
                                         int                n,
                                         int                nnz,
                                         T                  alpha,
                                         const int*         csr_row_ptr,
                                         const int*         csr_col_ind,
                                         const T*           csr_val,
                                         const T*           x,
                                         T                  beta,
                                         T*                 y,
                                         csrmv_algorithm    alg,
                                         const csrmv_descr* descr)
    {
        // std::cout << "alg: " << csrmv_alg_to_string(alg) << std::endl;

        switch(alg)
        {
        case csrmv_algorithm::default_algorithm:
        case csrmv_algorithm::rowsplit:
            csrmv_row_split_dispatch(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, descr);
            break;
        case csrmv_algorithm::nnzsplit:
            csrmv_nnz_split_dispatch(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, descr);
            break;
        case csrmv_algorithm::merge_path:
            csrmv_merge_path_dispatch(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, descr);
            break;
        case csrmv_algorithm::lrb:
            csrmv_lrb_dispatch(
                m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, descr);
            break;
        default:
            throw std::runtime_error("Unknown csrmv_algorithm");
        }
    }
}

//-------------------------------------------------------------------------------
// Compute residual res = b - A * x
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_compute_residual(int        m,
                                   int        n,
                                   int        nnz,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   const T*   csr_val,
                                   const T*   x,
                                   const T*   b,
                                   T*         res)
{
    ROUTINE_TRACE("linalg::cuda_compute_residual_impl");
    compute_residual_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, x, b, res);
    CHECK_CUDA_LAUNCH_ERROR();
}

void linalg::free_csrmv_cuda_data(csrmv_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->bin_start_ptr != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->bin_start_ptr));
            descr->bin_start_ptr = nullptr;
        }
        if(descr->row_index_in_bin != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->row_index_in_bin));
            descr->row_index_in_bin = nullptr;
        }
        if(descr->row_index_in_bin_sorted != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->row_index_in_bin_sorted));
            descr->row_index_in_bin_sorted = nullptr;
        }
    }
}

template <typename T>
void linalg::cuda_csrmv_analysis(int             m,
                                 int             n,
                                 int             nnz,
                                 const int*      csr_row_ptr,
                                 const int*      csr_col_ind,
                                 const T*        csr_val,
                                 csrmv_algorithm alg,
                                 csrmv_descr*    descr)
{
    // Free cuda memory that may have been allocated from previous calls to analysis
    free_csrmv_cuda_data(descr);

    csrmv_analysis_algorithm_dispatch(m, n, nnz, csr_row_ptr, csr_col_ind, alg, descr);
}

template <typename T>
void linalg::cuda_csrmv_solve(int                m,
                              int                n,
                              int                nnz,
                              T                  alpha,
                              const int*         csr_row_ptr,
                              const int*         csr_col_ind,
                              const T*           csr_val,
                              const T*           x,
                              T                  beta,
                              T*                 y,
                              csrmv_algorithm    alg,
                              const csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::cuda_csrmv_solve");

    csrmv_algorithm_dispatch(
        m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y, alg, descr);
}

template void linalg::cuda_compute_residual<double>(
    int, int, int, const int*, const int*, const double*, const double*, const double*, double*);
template void linalg::cuda_compute_residual<float>(
    int, int, int, const int*, const int*, const float*, const float*, const float*, float*);
template void linalg::cuda_csrmv_analysis<double>(
    int, int, int, const int*, const int*, const double*, csrmv_algorithm, csrmv_descr*);
template void linalg::cuda_csrmv_analysis<float>(
    int, int, int, const int*, const int*, const float*, csrmv_algorithm, csrmv_descr*);
template void linalg::cuda_csrmv_solve<double>(int,
                                               int,
                                               int,
                                               double,
                                               const int*,
                                               const int*,
                                               const double*,
                                               const double*,
                                               double,
                                               double*,
                                               csrmv_algorithm,
                                               const csrmv_descr*);
template void linalg::cuda_csrmv_solve<float>(int,
                                              int,
                                              int,
                                              float,
                                              const int*,
                                              const int*,
                                              const float*,
                                              const float*,
                                              float,
                                              float*,
                                              csrmv_algorithm,
                                              const csrmv_descr*);
