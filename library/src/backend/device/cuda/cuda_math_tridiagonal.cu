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
#include <map>

#include "cuda_math.h"

#include "tridiagonal_cyclic_reduction_kernels.cuh"
#include "tridiagonal_solver_kernels.cuh"
#include "tridiagonal_thomas_algorithm_kernels.cuh"
#include "tridiagonal_tiled_pcr_spike_kernels.cuh"

struct linalg::tridiagonal_descr
{
    float* lower_modified;
    float* main_modified;
    float* upper_modified;
    float* b_modified;

    float* spike_lower;
    float* spike_main;
    float* spike_upper;
    float* spike_b;
    float* spike_x;
};

void linalg::allocate_tridiagonal_cuda_data(tridiagonal_descr* descr)
{
    descr->lower_modified = nullptr;
    descr->main_modified  = nullptr;
    descr->upper_modified = nullptr;
    descr->b_modified     = nullptr;

    descr->spike_lower = nullptr;
    descr->spike_main  = nullptr;
    descr->spike_upper = nullptr;
    descr->spike_b     = nullptr;
    descr->spike_x     = nullptr;
}

void linalg::free_tridiagonal_cuda_data(tridiagonal_descr* descr)
{
    if(descr->lower_modified != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->lower_modified));
        descr->lower_modified = nullptr;
    }
    if(descr->main_modified != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->main_modified));
        descr->main_modified = nullptr;
    }
    if(descr->upper_modified != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->upper_modified));
        descr->upper_modified = nullptr;
    }
    if(descr->b_modified != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->b_modified));
        descr->b_modified = nullptr;
    }

    if(descr->spike_lower != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->spike_lower));
        descr->spike_lower = nullptr;
    }
    if(descr->spike_main != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->spike_main));
        descr->spike_main = nullptr;
    }
    if(descr->spike_upper != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->spike_upper));
        descr->spike_upper = nullptr;
    }
    if(descr->spike_b != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->spike_b));
        descr->spike_b = nullptr;
    }
    if(descr->spike_x != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->spike_x));
        descr->spike_x = nullptr;
    }
}

void linalg::cuda_tridiagonal_analysis(int                m,
                                       int                n,
                                       const float*       lower_diag,
                                       const float*       main_diag,
                                       const float*       upper_diag,
                                       tridiagonal_descr* descr)
{
    // Re-analysis with different dimensions must release old buffers first.
    free_tridiagonal_cuda_data(descr);

    constexpr int BLOCKSIZE = 256;
    int           nblocks   = ((m - 1) / BLOCKSIZE + 1);

    CHECK_CUDA(cudaMalloc((void**)&descr->lower_modified, sizeof(float) * m));
    CHECK_CUDA(cudaMalloc((void**)&descr->main_modified, sizeof(float) * m));
    CHECK_CUDA(cudaMalloc((void**)&descr->upper_modified, sizeof(float) * m));
    CHECK_CUDA(cudaMalloc((void**)&descr->b_modified, sizeof(float) * m * n));

    CHECK_CUDA(cudaMalloc((void**)&descr->spike_lower, sizeof(float) * 2 * nblocks));
    CHECK_CUDA(cudaMalloc((void**)&descr->spike_main, sizeof(float) * 2 * nblocks));
    CHECK_CUDA(cudaMalloc((void**)&descr->spike_upper, sizeof(float) * 2 * nblocks));
    CHECK_CUDA(cudaMalloc((void**)&descr->spike_b, sizeof(float) * 2 * nblocks * n));
    CHECK_CUDA(cudaMalloc((void**)&descr->spike_x, sizeof(float) * 2 * nblocks * n));
}

namespace linalg
{
    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    static void launch_pcr_tiled_forward_elimination_kernel(int      m,
                                                            int      n,
                                                            const T* lower,
                                                            const T* main,
                                                            const T* upper,
                                                            const T* B,
                                                            T*       lower_modified,
                                                            T*       main_modified,
                                                            T*       upper_modified,
                                                            T*       B_modified,
                                                            T*       spike_lower,
                                                            T*       spike_main,
                                                            T*       spike_upper,
                                                            T*       spike_B)
    {
        dim3 grid((m - 1) / BLOCKSIZE + 1, (n - 1) / NUM_RHS + 1);
        dim3 block(BLOCKSIZE);

        pcr_tiled_forward_kernel<BLOCKSIZE, NUM_RHS><<<grid, block>>>(m,
                                                                      n,
                                                                      lower,
                                                                      main,
                                                                      upper,
                                                                      B,
                                                                      lower_modified,
                                                                      main_modified,
                                                                      upper_modified,
                                                                      B_modified,
                                                                      spike_lower,
                                                                      spike_main,
                                                                      spike_upper,
                                                                      spike_B);
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    static void launch_spike_solver_pcr_kernel(int      num_spikes,
                                               int      n,
                                               const T* l_spike,
                                               const T* m_spike,
                                               const T* u_spike,
                                               const T* B_spike,
                                               T*       X_spike_out)
    {
        spike_solver_pcr_kernel<BLOCKSIZE, NUM_RHS>
            <<<dim3((n - 1) / NUM_RHS + 1), dim3(BLOCKSIZE)>>>(
                num_spikes, n, l_spike, m_spike, u_spike, B_spike, X_spike_out);
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    static void launch_pcr_tiled_backward_substitution_kernel(int      m,
                                                              int      n,
                                                              int      num_spikes,
                                                              const T* lower_modified,
                                                              const T* main_modified,
                                                              const T* upper_modified,
                                                              const T* B_modified,
                                                              const T* X_spike,
                                                              T*       X_final)
    {
        dim3 grid((m - 1) / BLOCKSIZE + 1, (n - 1) / NUM_RHS + 1);
        dim3 block(BLOCKSIZE);

        pcr_tiled_backward_kernel<BLOCKSIZE, NUM_RHS><<<grid, block>>>(m,
                                                                       n,
                                                                       num_spikes,
                                                                       lower_modified,
                                                                       main_modified,
                                                                       upper_modified,
                                                                       B_modified,
                                                                       X_spike,
                                                                       X_final);
    }

    template <typename T>
    static void tridiagonal_tile_pcr_spike_solver(int                      m,
                                                  int                      n,
                                                  const T*                 lower_diag,
                                                  const T*                 main_diag,
                                                  const T*                 upper_diag,
                                                  const T*                 b,
                                                  T*                       x,
                                                  const tridiagonal_descr* descr)
    {
        constexpr int BLOCKSIZE  = 256; // remember to change in analysis as well!
        constexpr int NUM_RHS    = 8;
        int           nblocks    = ((m - 1) / BLOCKSIZE + 1);
        int           num_spikes = 2 * nblocks;

        launch_pcr_tiled_forward_elimination_kernel<BLOCKSIZE, NUM_RHS>(m,
                                                                        n,
                                                                        lower_diag,
                                                                        main_diag,
                                                                        upper_diag,
                                                                        b,
                                                                        descr->lower_modified,
                                                                        descr->main_modified,
                                                                        descr->upper_modified,
                                                                        descr->b_modified,
                                                                        descr->spike_lower,
                                                                        descr->spike_main,
                                                                        descr->spike_upper,
                                                                        descr->spike_b);

        using spike_solver_pcr_launch_ptr
            = void (*)(int, int, const T*, const T*, const T*, const T*, T*);

        static const std::map<int, spike_solver_pcr_launch_ptr> k_spike_solver_dispatch = {
            {4, launch_spike_solver_pcr_kernel<4, NUM_RHS, T>},
            {8, launch_spike_solver_pcr_kernel<8, NUM_RHS, T>},
            {16, launch_spike_solver_pcr_kernel<16, NUM_RHS, T>},
            {32, launch_spike_solver_pcr_kernel<32, NUM_RHS, T>},
            {64, launch_spike_solver_pcr_kernel<64, NUM_RHS, T>},
            {128, launch_spike_solver_pcr_kernel<128, NUM_RHS, T>},
            {256, launch_spike_solver_pcr_kernel<256, NUM_RHS, T>},
            {512, launch_spike_solver_pcr_kernel<512, NUM_RHS, T>},
            {1024, launch_spike_solver_pcr_kernel<1024, NUM_RHS, T>},
        };

        auto dispatch_it = k_spike_solver_dispatch.lower_bound(num_spikes);
        if(dispatch_it != k_spike_solver_dispatch.end())
        {
            dispatch_it->second(num_spikes,
                                n,
                                descr->spike_lower,
                                descr->spike_main,
                                descr->spike_upper,
                                descr->spike_b,
                                descr->spike_x);
        }

        launch_pcr_tiled_backward_substitution_kernel<BLOCKSIZE, NUM_RHS>(m,
                                                                          n,
                                                                          num_spikes,
                                                                          descr->lower_modified,
                                                                          descr->main_modified,
                                                                          descr->upper_modified,
                                                                          descr->b_modified,
                                                                          descr->spike_x,
                                                                          x);
    }

    template <uint32_t BLOCKSIZE, uint32_t M, typename T>
    static void launch_thomas_algorithm_kernel(
        int n, const T* lower_diag, const T* main_diag, const T* upper_diag, const T* B, T* X)
    {
        thomas_algorithm_kernel<BLOCKSIZE, M>
            <<<((n - 1) / BLOCKSIZE + 1), BLOCKSIZE>>>(n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void tridiagonal_thomas_algorithm_solver(int      m,
                                                    int      n,
                                                    const T* lower_diag,
                                                    const T* main_diag,
                                                    const T* upper_diag,
                                                    const T* B,
                                                    T*       X)
    {
        using thomas_algorithm_launch_ptr
            = void (*)(int, const T*, const T*, const T*, const T*, T*);

        static const std::map<int, thomas_algorithm_launch_ptr> k_thomas_algorithm_dispatch = {
            {2, launch_thomas_algorithm_kernel<256, 2, T>},
            {3, launch_thomas_algorithm_kernel<256, 3, T>},
            {4, launch_thomas_algorithm_kernel<256, 4, T>},
            {5, launch_thomas_algorithm_kernel<256, 5, T>},
            {6, launch_thomas_algorithm_kernel<256, 6, T>},
            {7, launch_thomas_algorithm_kernel<256, 7, T>},
            {8, launch_thomas_algorithm_kernel<256, 8, T>},
            {9, launch_thomas_algorithm_kernel<256, 9, T>},
            {10, launch_thomas_algorithm_kernel<256, 10, T>},
        };

        auto dispatch_it = k_thomas_algorithm_dispatch.find(m);
        if(dispatch_it != k_thomas_algorithm_dispatch.end())
        {
            dispatch_it->second(n, lower_diag, main_diag, upper_diag, B, X);
        }
    }

    template <typename T>
    static void launch_tridiagonal_m16_kernel(int      m,
                                              int      n,
                                              const T* lower_diag,
                                              const T* main_diag,
                                              const T* upper_diag,
                                              const T* b,
                                              T*       x)
    {
        constexpr int BLOCKSIZE = 256;
        constexpr int WARP_SIZE = 16;
        thomas_pcr_wavefront_kernel<BLOCKSIZE, WARP_SIZE>
            <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
                m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m32_kernel(int      m,
                                              int      n,
                                              const T* lower_diag,
                                              const T* main_diag,
                                              const T* upper_diag,
                                              const T* b,
                                              T*       x)
    {
        constexpr int BLOCKSIZE = 256;
        constexpr int WARP_SIZE = 16;
        constexpr int M         = 32;
        thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
            <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
                m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m64_kernel(int      m,
                                              int      n,
                                              const T* lower_diag,
                                              const T* main_diag,
                                              const T* upper_diag,
                                              const T* b,
                                              T*       x)
    {
        constexpr int BLOCKSIZE = 256;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 64;
        thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
            <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
                m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m128_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* b,
                                               T*       x)
    {
        constexpr int BLOCKSIZE = 128;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 128;

        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m256_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* b,
                                               T*       x)
    {
        crpcr_pow2_shared_multi_rhs_kernel<128, 64, 8>
            <<<((n - 1) / 8 + 1), 128>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m512_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* b,
                                               T*       x)
    {
        crpcr_pow2_shared_multi_rhs_kernel<256, 128, 8>
            <<<((n - 1) / 8 + 1), 256>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void launch_tridiagonal_m1024_kernel(int      m,
                                                int      n,
                                                const T* lower_diag,
                                                const T* main_diag,
                                                const T* upper_diag,
                                                const T* b,
                                                T*       x)
    {
        crpcr_pow2_shared_multi_rhs_kernel<512, 256, 8>
            <<<((n - 1) / 8 + 1), 512>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
    }

    template <typename T>
    static void tridiagonal_pcr_solver_dispatch(int      m,
                                                int      n,
                                                const T* lower_diag,
                                                const T* main_diag,
                                                const T* upper_diag,
                                                const T* b,
                                                T*       x)
    {
        using midrange_launch_ptr = void (*)(int, int, const T*, const T*, const T*, const T*, T*);

        static const std::map<int, midrange_launch_ptr> k_midrange_dispatch = {
            {16, launch_tridiagonal_m16_kernel<T>},
            {32, launch_tridiagonal_m32_kernel<T>},
            {64, launch_tridiagonal_m64_kernel<T>},
            {128, launch_tridiagonal_m128_kernel<T>},
            {256, launch_tridiagonal_m256_kernel<T>},
            {512, launch_tridiagonal_m512_kernel<T>},
            {1024, launch_tridiagonal_m1024_kernel<T>},
        };

        auto dispatch_it = k_midrange_dispatch.lower_bound(m);
        if(dispatch_it != k_midrange_dispatch.end())
        {
            dispatch_it->second(m, n, lower_diag, main_diag, upper_diag, b, x);
        }
    }
}

void linalg::cuda_tridiagonal_solver(int                      m,
                                     int                      n,
                                     const float*             lower_diag,
                                     const float*             main_diag,
                                     const float*             upper_diag,
                                     const float*             b,
                                     float*                   x,
                                     const tridiagonal_descr* descr)
{
    if(m <= 10)
    {
        tridiagonal_thomas_algorithm_solver(m, n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m <= 1024)
    {
        tridiagonal_pcr_solver_dispatch(m, n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m <= 131072)
    {
        tridiagonal_tile_pcr_spike_solver(m, n, lower_diag, main_diag, upper_diag, b, x, descr);
    }
    else
    {
        std::cerr << "Error: cuda_tridiagonal_solver only supports m = 2 to 131072." << std::endl;
        return;
    }

    CHECK_CUDA_LAUNCH_ERROR();
}
