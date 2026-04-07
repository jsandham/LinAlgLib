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

#include <Vector>
#include <iostream>
#include <map>

#include "linalg_enums.h"

#include "cuda_tridiagonal.h"

#include "tridiagonal_cyclic_reduction_kernels.cuh"
#include "tridiagonal_solver_kernels.cuh"
#include "tridiagonal_thomas_algorithm_kernels.cuh"
#include "tridiagonal_tiled_pcr_spike_kernels.cuh"

#include "tridiagonal_spike_kernels.cuh"

static constexpr int MAX_RECURSION_LEVELS = 3;

struct linalg::tridiagonal_descr
{
    pivoting_strategy pivoting_strategy;

    // Buffers for non-pivoting approach (one per recursion level)
    float* lower_modified[MAX_RECURSION_LEVELS];
    float* main_modified[MAX_RECURSION_LEVELS];
    float* upper_modified[MAX_RECURSION_LEVELS];
    float* b_modified[MAX_RECURSION_LEVELS];

    float* spike_lower[MAX_RECURSION_LEVELS];
    float* spike_main[MAX_RECURSION_LEVELS];
    float* spike_upper[MAX_RECURSION_LEVELS];
    float* spike_b[MAX_RECURSION_LEVELS];
    float* spike_x[MAX_RECURSION_LEVELS];

    // Buffers for partial pivoting approach (to be implemented)
    float* lower_pad;
    float* main_pad;
    float* upper_pad;

    float* w_pad;
    float* v_pad;
};

void linalg::free_tridiagonal_cuda_data(tridiagonal_descr* descr)
{
    for(int level = 0; level < MAX_RECURSION_LEVELS; level++)
    {
        if(descr->lower_modified[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->lower_modified[level]));
            descr->lower_modified[level] = nullptr;
        }
        if(descr->main_modified[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->main_modified[level]));
            descr->main_modified[level] = nullptr;
        }
        if(descr->upper_modified[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->upper_modified[level]));
            descr->upper_modified[level] = nullptr;
        }
        if(descr->b_modified[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->b_modified[level]));
            descr->b_modified[level] = nullptr;
        }

        if(descr->spike_lower[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_lower[level]));
            descr->spike_lower[level] = nullptr;
        }
        if(descr->spike_main[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_main[level]));
            descr->spike_main[level] = nullptr;
        }
        if(descr->spike_upper[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_upper[level]));
            descr->spike_upper[level] = nullptr;
        }
        if(descr->spike_b[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_b[level]));
            descr->spike_b[level] = nullptr;
        }
        if(descr->spike_x[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_x[level]));
            descr->spike_x[level] = nullptr;
        }
    }

    if(descr->lower_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->lower_pad));
        descr->lower_pad = nullptr;
    }
    if(descr->main_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->main_pad));
        descr->main_pad = nullptr;
    }
    if(descr->upper_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->upper_pad));
        descr->upper_pad = nullptr;
    }

    if(descr->w_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->w_pad));
        descr->w_pad = nullptr;
    }
    if(descr->v_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->v_pad));
        descr->v_pad = nullptr;
    }
}

namespace linalg
{
    static uint64_t next_power_of_two(uint64_t m)
    {
        // If m is already a power of 2 or 0, return m (or 1 if you prefer 2^0)
        if(m == 0)
            return 1;

        // Decrement m so that if it is already a power of 2,
        // the operations below don't jump it to the next one.
        m--;

        // Fill all bits to the right of the most significant bit with 1s
        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        m |= m >> 16;
        m |= m >> 32; // Include this if using 64-bit integers

        // Adding 1 results in a single bit set at the next power of 2
        return m + 1;
    }

    static void tridiagonal_nonpivoting_analysis_dispatch(int                m,
                                                          int                n,
                                                          const float*       lower_diag,
                                                          const float*       main_diag,
                                                          const float*       upper_diag,
                                                          tridiagonal_descr* descr)
    {
        constexpr int BLOCKSIZE = 256;

        int current_m = m;
        for(int level = 0; level < MAX_RECURSION_LEVELS; level++)
        {
            if(current_m <= 1024)
                break;

            int nblocks    = ((current_m - 1) / BLOCKSIZE + 1);
            int num_spikes = 2 * nblocks;

            CHECK_CUDA(cudaMalloc((void**)&descr->lower_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(cudaMalloc((void**)&descr->main_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(cudaMalloc((void**)&descr->upper_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(cudaMalloc((void**)&descr->b_modified[level], sizeof(float) * current_m * n));

            CHECK_CUDA(cudaMalloc((void**)&descr->spike_lower[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_main[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_upper[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_b[level], sizeof(float) * num_spikes * n));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_x[level], sizeof(float) * num_spikes * n));

            current_m = num_spikes;
        }
    }

    static void tridiagonal_partial_pivoting_analysis_dispatch(int                m,
                                                               int                n,
                                                               const float*       lower_diag,
                                                               const float*       main_diag,
                                                               const float*       upper_diag,
                                                               tridiagonal_descr* descr)
    {
        int m_pad = next_power_of_two(m);

        CHECK_CUDA(cudaMalloc((void**)&descr->lower_pad, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->main_pad, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->upper_pad, sizeof(float) * m_pad));

        CHECK_CUDA(cudaMalloc((void**)&descr->w_pad, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->v_pad, sizeof(float) * m_pad));
    }

    static void tridiagonal_nonpivoting_solver_dispatch(int                      m,
                                                        int                      n,
                                                        const float*             lower_diag,
                                                        const float*             main_diag,
                                                        const float*             upper_diag,
                                                        const float*             b,
                                                        float*                   x,
                                                        const tridiagonal_descr* descr,
                                                        int                      level = 0);
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

    switch(descr->pivoting_strategy)
    {
    case pivoting_strategy::none:
        tridiagonal_nonpivoting_analysis_dispatch(m, n, lower_diag, main_diag, upper_diag, descr);
        break;
    case pivoting_strategy::partial:
        tridiagonal_partial_pivoting_analysis_dispatch(
            m, n, lower_diag, main_diag, upper_diag, descr);
        break;
    }
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
                                                  const tridiagonal_descr* descr,
                                                  int                      level)
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
                                                                        descr->lower_modified[level],
                                                                        descr->main_modified[level],
                                                                        descr->upper_modified[level],
                                                                        descr->b_modified[level],
                                                                        descr->spike_lower[level],
                                                                        descr->spike_main[level],
                                                                        descr->spike_upper[level],
                                                                        descr->spike_b[level]);

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
                                descr->spike_lower[level],
                                descr->spike_main[level],
                                descr->spike_upper[level],
                                descr->spike_b[level],
                                descr->spike_x[level]);
        }
        else
        {
            tridiagonal_nonpivoting_solver_dispatch(num_spikes,
                                                    n,
                                                    descr->spike_lower[level],
                                                    descr->spike_main[level],
                                                    descr->spike_upper[level],
                                                    descr->spike_b[level],
                                                    descr->spike_x[level],
                                                    descr,
                                                    level + 1);
        }

        launch_pcr_tiled_backward_substitution_kernel<BLOCKSIZE, NUM_RHS>(m,
                                                                          n,
                                                                          num_spikes,
                                                                          descr->lower_modified[level],
                                                                          descr->main_modified[level],
                                                                          descr->upper_modified[level],
                                                                          descr->b_modified[level],
                                                                          descr->spike_x[level],
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

        // Something wrong with the thomas_pcr_wavefront_kernel2 kernel. Fails in debug but passes in release
        // constexpr int BLOCKSIZE = 256;
        // constexpr int WARP_SIZE = 16;
        // constexpr int M         = 32;
        // thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
        // <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
        // m, n, lower_diag, main_diag, upper_diag, b, x);
        constexpr int BLOCKSIZE = 32;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 32;
        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
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
        // Something wrong with the thomas_pcr_wavefront_kernel2 kernel. Fails in debug but passes in release
        // constexpr int BLOCKSIZE = 256;
        // constexpr int WARP_SIZE = 32;
        // constexpr int M         = 64;
        //thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
        //    <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
        //        m, n, lower_diag, main_diag, upper_diag, b, x);
        constexpr int BLOCKSIZE = 64;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 64;
        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
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

    static void tridiagonal_nonpivoting_solver_dispatch(int                      m,
                                                        int                      n,
                                                        const float*             lower_diag,
                                                        const float*             main_diag,
                                                        const float*             upper_diag,
                                                        const float*             b,
                                                        float*                   x,
                                                        const tridiagonal_descr* descr,
                                                        int                      level)
    {
        // std::cout << "tridiagonal_nonpivoting_solver_dispatch called with m: " << m << ", n: " << n
        //           << ", level: " << level << std::endl;
        if(m <= 10)
        {
            tridiagonal_thomas_algorithm_solver(m, n, lower_diag, main_diag, upper_diag, b, x);
        }
        else if(m <= 1024)
        {
            tridiagonal_pcr_solver_dispatch(m, n, lower_diag, main_diag, upper_diag, b, x);
        }
        else
        {
            tridiagonal_tile_pcr_spike_solver(m, n, lower_diag, main_diag, upper_diag, b, x, descr, level);
        }

        CHECK_CUDA_LAUNCH_ERROR();
    }

    static void tridiagonal_partial_pivoting_solver_dispatch(int                      m,
                                                             int                      n,
                                                             const float*             lower_diag,
                                                             const float*             main_diag,
                                                             const float*             upper_diag,
                                                             const float*             b,
                                                             float*                   x,
                                                             const tridiagonal_descr* descr)
    {
        std::vector<float> h_lower(m);
        std::vector<float> h_main(m);
        std::vector<float> h_upper(m);
        std::vector<float> h_b(m * n);
        CHECK_CUDA(
            cudaMemcpy(h_lower.data(), lower_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_main.data(), main_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(h_upper.data(), upper_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_b.data(), b, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

        std::cout << "h_main" << std::endl;
        for(int i = 0; i < m; i++)
        {
            std::cout << h_main[i] << " ";
        }
        std::cout << "" << std::endl;

        int m_pad = next_power_of_two(m);

        std::cout << "m: " << m << ", m_pad: " << m_pad << std::endl;

        data_marshaling_kernel<1024, 32><<<(m - 1) / 1024 + 1, 1024>>>(m,
                                                                       m_pad,
                                                                       lower_diag,
                                                                       main_diag,
                                                                       upper_diag,
                                                                       descr->lower_pad,
                                                                       descr->main_pad,
                                                                       descr->upper_pad);
        CHECK_CUDA_LAUNCH_ERROR();

        std::vector<float> h_lower_pad(m_pad);
        std::vector<float> h_main_pad(m_pad);
        std::vector<float> h_upper_pad(m_pad);
        CHECK_CUDA(cudaMemcpy(h_lower_pad.data(),
                              descr->lower_pad,
                              sizeof(float) * m_pad,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_main_pad.data(),
                              descr->main_pad,
                              sizeof(float) * m_pad,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_upper_pad.data(),
                              descr->upper_pad,
                              sizeof(float) * m_pad,
                              cudaMemcpyDeviceToHost));
        std::cout << "h_lower_pad" << std::endl;
        for(int i = 0; i < m_pad; i++)
        {
            std::cout << h_lower_pad[i] << " ";
        }
        std::cout << "" << std::endl;
        std::cout << "h_main_pad" << std::endl;
        for(int i = 0; i < m_pad; i++)
        {
            std::cout << h_main_pad[i] << " ";
        }
        std::cout << "" << std::endl;
        std::cout << "h_upper_pad" << std::endl;
        for(int i = 0; i < m_pad; i++)
        {
            std::cout << h_upper_pad[i] << " ";
        }
        std::cout << "" << std::endl;

        CHECK_CUDA(cudaMemset(descr->w_pad, 0, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMemset(descr->v_pad, 0, sizeof(float) * m_pad));

        //     constexpr uint32_t BLOCKSIZE = 8;

        //     int block_dim = 2;
        //     int m_pad     = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
        //     int gridsize  = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);

        //     while(gridsize > 16)
        //     {
        //         block_dim *= 2;
        //         m_pad    = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
        //         gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
        //     }
        //     // round up to next power of 2
        //     //gridsize = fnp2(gridsize);

        //     std::cout << "gridsize: " << gridsize << ", block_dim: " << block_dim << ", m_pad: " << m_pad
        //               << std::endl;

        //     float* db_pad = nullptr;
        //     CHECK_CUDA(cudaMalloc((void**)&db_pad, sizeof(float) * m_pad * n));

        //     //Call transpose kernel
        //     dim3          grid(((m_pad - 1) / BLOCKSIZE + 1), n);
        //     dim3          block(BLOCKSIZE);
        //     if(block_dim == 2)
        //     {
        //         transpose_and_pad_array_shared_kernel<BLOCKSIZE, 2><<<grid, block>>>(m, m_pad, m, b, db_pad, 0.0f);
        //     }
        //     else if(block_dim == 4)
        //     {
        //         transpose_and_pad_array_shared_kernel<BLOCKSIZE, 4><<<grid, block>>>(m, m_pad, m, b, db_pad, 0.0f);
        //     }

        //     // Copy b back to host for pivoting and factorization
        //     std::vector<float> hb_pad(m_pad * n);
        //     CHECK_CUDA(cudaMemcpy(hb_pad.data(), db_pad, sizeof(float) * m_pad * n, cudaMemcpyDeviceToHost));

        //     std::cout << "hb_pad after transpose" << std::endl;
        //     for(int i = 0; i < m_pad * n; i++)
        //     {
        //         std::cout << hb_pad[i] << " ";
        //     }
        //     std::cout << "" << std::endl;

        //     CHECK_CUDA(cudaFree(db_pad));

        //     CHECK_CUDA_LAUNCH_ERROR();
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
    switch(descr->pivoting_strategy)
    {
    case pivoting_strategy::none:
        tridiagonal_nonpivoting_solver_dispatch(
            m, n, lower_diag, main_diag, upper_diag, b, x, descr);
        break;
    case pivoting_strategy::partial:
        tridiagonal_partial_pivoting_solver_dispatch(
            m, n, lower_diag, main_diag, upper_diag, b, x, descr);
        break;
    }
}
