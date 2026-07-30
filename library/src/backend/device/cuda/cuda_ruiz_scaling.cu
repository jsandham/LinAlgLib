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

#include <cuda_runtime.h>

#include "cuda_ruiz_scaling.h"

#include "ruiz_scaling_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// Ruiz scaling
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_ruiz_scaling(T*         D1,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               T*         csr_val,
                               int        m,
                               T*         D2,
                               int        max_k,
                               T          tol)
{
    ROUTINE_TRACE("linalg::cuda_ruiz_scaling_impl");
}

template <typename T>
void linalg::cuda_symmetric_ruiz_scaling(
    T* D, const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int m, int max_k, T tol)
{
    ROUTINE_TRACE("linalg::cuda_symmetric_ruiz_scaling_impl");

    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));
    CHECK_CUDA(cudaMemset((void**)&workspace, 0, sizeof(T) * 256));

    T* dmax_divergence = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dmax_divergence, sizeof(T)));

    // D^0 = I
    set_D_to_one_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, D);
    CHECK_CUDA_LAUNCH_ERROR();

    // DR = DC
    T* DR = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&DR, sizeof(T) * m));

    for(int iter = 0; iter < max_k; iter++)
    {
        CHECK_CUDA(cudaMemset((void**)&DR, 0, sizeof(T) * m));

        fill_DR_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m, csr_row_ptr, csr_val, DR);
        CHECK_CUDA_LAUNCH_ERROR();

        const T eps = std::numeric_limits<T>::epsilon();

        compute_max_divergence_part1<256><<<256, 256>>>(m, eps, DR, workspace);
        CHECK_CUDA_LAUNCH_ERROR();

        compute_max_divergence_part2<256><<<1, 256>>>(m, workspace, dmax_divergence);
        CHECK_CUDA_LAUNCH_ERROR();

        T max_divergence = static_cast<T>(0);
        CHECK_CUDA(cudaMemcpy(&max_divergence, dmax_divergence, sizeof(T), cudaMemcpyDeviceToHost));

        std::cout << "max_divergence: " << max_divergence << std::endl;

        if(max_divergence < tol)
        {
            break;
        }

        // Update A_k+1 = DR^-1 * A_k * DR^-1
        update_A<256, 32>
            <<<((m - 1) / (256 / 32) + 1), 256>>>(m, csr_row_ptr, csr_col_ind, csr_val, DR);
        CHECK_CUDA_LAUNCH_ERROR();

        // Update D_k+1 = D_k * DR^-1
        update_D<256><<<((m - 1) / 256 + 1), 256>>>(m, DR, D);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    CHECK_CUDA(cudaFree(DR));
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(dmax_divergence));
}

template void linalg::cuda_ruiz_scaling<double>(double*    D1,
                                                const int* csr_row_ptr,
                                                const int* csr_col_ind,
                                                double*    csr_val,
                                                int        m,
                                                double*    D2,
                                                int        max_k,
                                                double     tol);
template void linalg::cuda_symmetric_ruiz_scaling<double>(double*    D,
                                                          const int* csr_row_ptr,
                                                          const int* csr_col_ind,
                                                          double*    csr_val,
                                                          int        m,
                                                          int        max_k,
                                                          double     tol);
