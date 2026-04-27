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
    float* B_modified[MAX_RECURSION_LEVELS];

    float* spike_lower[MAX_RECURSION_LEVELS];
    float* spike_main[MAX_RECURSION_LEVELS];
    float* spike_upper[MAX_RECURSION_LEVELS];
    float* spike_B[MAX_RECURSION_LEVELS];
    float* spike_X[MAX_RECURSION_LEVELS];

    // Buffers for partial pivoting approach (to be implemented)
    float* lower_pad;
    float* main_pad;
    float* upper_pad;
    float* B_pad;

    float* w_pad;
    float* v_pad;

    int*   pivot;
    float* mt;
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
        if(descr->B_modified[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->B_modified[level]));
            descr->B_modified[level] = nullptr;
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
        if(descr->spike_B[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_B[level]));
            descr->spike_B[level] = nullptr;
        }
        if(descr->spike_X[level] != nullptr)
        {
            CHECK_CUDA(cudaFree(descr->spike_X[level]));
            descr->spike_X[level] = nullptr;
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
    if(descr->B_pad != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->B_pad));
        descr->B_pad = nullptr;
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
    if(descr->pivot != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->pivot));
        descr->pivot = nullptr;
    }
    if(descr->mt != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->mt));
        descr->mt = nullptr;
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

            CHECK_CUDA(
                cudaMalloc((void**)&descr->lower_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(cudaMalloc((void**)&descr->main_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(
                cudaMalloc((void**)&descr->upper_modified[level], sizeof(float) * current_m));
            CHECK_CUDA(
                cudaMalloc((void**)&descr->B_modified[level], sizeof(float) * current_m * n));

            CHECK_CUDA(cudaMalloc((void**)&descr->spike_lower[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_main[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_upper[level], sizeof(float) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_B[level], sizeof(float) * num_spikes * n));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_X[level], sizeof(float) * num_spikes * n));

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
        CHECK_CUDA(cudaMalloc((void**)&descr->B_pad, sizeof(float) * m_pad * n));

        CHECK_CUDA(cudaMalloc((void**)&descr->w_pad, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->v_pad, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->pivot, sizeof(int) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->mt, sizeof(float) * m_pad));
    }

    static void tridiagonal_nonpivoting_solver_dispatch(int                      m,
                                                        int                      n,
                                                        const float*             lower_diag,
                                                        const float*             main_diag,
                                                        const float*             upper_diag,
                                                        const float*             B,
                                                        float*                   X,
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
                                                  const T*                 B,
                                                  T*                       X,
                                                  const tridiagonal_descr* descr,
                                                  int                      level)
    {
        constexpr int BLOCKSIZE  = 256; // remember to change in analysis as well!
        constexpr int NUM_RHS    = 8;
        int           nblocks    = ((m - 1) / BLOCKSIZE + 1);
        int           num_spikes = 2 * nblocks;

        launch_pcr_tiled_forward_elimination_kernel<BLOCKSIZE, NUM_RHS>(
            m,
            n,
            lower_diag,
            main_diag,
            upper_diag,
            B,
            descr->lower_modified[level],
            descr->main_modified[level],
            descr->upper_modified[level],
            descr->B_modified[level],
            descr->spike_lower[level],
            descr->spike_main[level],
            descr->spike_upper[level],
            descr->spike_B[level]);

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
                                descr->spike_B[level],
                                descr->spike_X[level]);
        }
        else
        {
            tridiagonal_nonpivoting_solver_dispatch(num_spikes,
                                                    n,
                                                    descr->spike_lower[level],
                                                    descr->spike_main[level],
                                                    descr->spike_upper[level],
                                                    descr->spike_B[level],
                                                    descr->spike_X[level],
                                                    descr,
                                                    level + 1);
        }

        launch_pcr_tiled_backward_substitution_kernel<BLOCKSIZE, NUM_RHS>(
            m,
            n,
            num_spikes,
            descr->lower_modified[level],
            descr->main_modified[level],
            descr->upper_modified[level],
            descr->B_modified[level],
            descr->spike_X[level],
            X);
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
                                              const T* B,
                                              T*       X)
    {
        constexpr int BLOCKSIZE = 256;
        constexpr int WARP_SIZE = 16;
        thomas_pcr_wavefront_kernel<BLOCKSIZE, WARP_SIZE>
            <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
                m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m32_kernel(int      m,
                                              int      n,
                                              const T* lower_diag,
                                              const T* main_diag,
                                              const T* upper_diag,
                                              const T* B,
                                              T*       X)
    {

        // Something wrong with the thomas_pcr_wavefront_kernel2 kernel. Fails in debug but passes in release
        // constexpr int BLOCKSIZE = 256;
        // constexpr int WARP_SIZE = 16;
        // constexpr int M         = 32;
        // thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
        // <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
        // m, n, lower_diag, main_diag, upper_diag, B, X);
        constexpr int BLOCKSIZE = 32;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 32;
        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m64_kernel(int      m,
                                              int      n,
                                              const T* lower_diag,
                                              const T* main_diag,
                                              const T* upper_diag,
                                              const T* B,
                                              T*       X)
    {
        // Something wrong with the thomas_pcr_wavefront_kernel2 kernel. Fails in debug but passes in release
        // constexpr int BLOCKSIZE = 256;
        // constexpr int WARP_SIZE = 32;
        // constexpr int M         = 64;
        //thomas_pcr_wavefront_kernel2<BLOCKSIZE, WARP_SIZE, M>
        //    <<<((n - 1) / (BLOCKSIZE / WARP_SIZE) + 1), BLOCKSIZE>>>(
        //        m, n, lower_diag, main_diag, upper_diag, B, X);
        constexpr int BLOCKSIZE = 64;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 64;
        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m128_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* B,
                                               T*       X)
    {
        constexpr int BLOCKSIZE = 128;
        constexpr int WARP_SIZE = 32;
        constexpr int M         = 128;

        pcr_shared_kernel2<BLOCKSIZE, WARP_SIZE, M, 8>
            <<<((n - 1) / 8 + 1), BLOCKSIZE>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m256_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* B,
                                               T*       X)
    {
        crpcr_pow2_shared_multi_rhs_kernel<128, 64, 8>
            <<<((n - 1) / 8 + 1), 128>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m512_kernel(int      m,
                                               int      n,
                                               const T* lower_diag,
                                               const T* main_diag,
                                               const T* upper_diag,
                                               const T* B,
                                               T*       X)
    {
        crpcr_pow2_shared_multi_rhs_kernel<256, 128, 8>
            <<<((n - 1) / 8 + 1), 256>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void launch_tridiagonal_m1024_kernel(int      m,
                                                int      n,
                                                const T* lower_diag,
                                                const T* main_diag,
                                                const T* upper_diag,
                                                const T* B,
                                                T*       X)
    {
        crpcr_pow2_shared_multi_rhs_kernel<512, 256, 8>
            <<<((n - 1) / 8 + 1), 512>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
    }

    template <typename T>
    static void tridiagonal_pcr_solver_dispatch(int      m,
                                                int      n,
                                                const T* lower_diag,
                                                const T* main_diag,
                                                const T* upper_diag,
                                                const T* B,
                                                T*       X)
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
            dispatch_it->second(m, n, lower_diag, main_diag, upper_diag, B, X);
        }
    }

    static void tridiagonal_nonpivoting_solver_dispatch(int                      m,
                                                        int                      n,
                                                        const float*             lower_diag,
                                                        const float*             main_diag,
                                                        const float*             upper_diag,
                                                        const float*             B,
                                                        float*                   X,
                                                        const tridiagonal_descr* descr,
                                                        int                      level)
    {
        // std::cout << "tridiagonal_nonpivoting_solver_dispatch called with m: " << m << ", n: " << n
        //           << ", level: " << level << std::endl;
        if(m <= 10)
        {
            tridiagonal_thomas_algorithm_solver(m, n, lower_diag, main_diag, upper_diag, B, X);
        }
        else if(m <= 1024)
        {
            tridiagonal_pcr_solver_dispatch(m, n, lower_diag, main_diag, upper_diag, B, X);
        }
        else
        {
            tridiagonal_tile_pcr_spike_solver(
                m, n, lower_diag, main_diag, upper_diag, B, X, descr, level);
        }

        CHECK_CUDA_LAUNCH_ERROR();
    }


    static void host_thomas_algorithm(int      m,
                                           int      n,
                                           const float* lower_diag,
                                           const float* main_diag,
                                           const float* upper_diag,
                                           const float* b,
                                           float*       x)
    {
        std::vector<float> c_prime(m);
        c_prime[0] = upper_diag[0] / main_diag[0];
        for(int i = 1; i < m - 1; i++)
        {
            float denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
            c_prime[i] = upper_diag[i] / denom;
        }

        for(int j = 0; j < n; j++)
        {
            std::vector<float> d_prime(m);
            d_prime[0] = b[m * j + 0] / main_diag[0];
            for(int i = 1; i < m; i++)
            {
                float num      = b[m * j + i] - lower_diag[i] * d_prime[i - 1];
                float denom    = main_diag[i] - lower_diag[i] * c_prime[i - 1];
                d_prime[i] = num / denom;
            }
            x[m * j + (m - 1)] = d_prime[m - 1];
            for(int i = m - 2; i >= 0; i--)
                x[m * j + i] = d_prime[i] - c_prime[i] * x[m * j + (i + 1)];
        }
    }

    #define DEBUG_PRINT_ARRAY(arr, size, name) \
        do { \
            std::cout << name << ": "; \
            for(int i = 0; i < size; i++) { \
                std::cout << arr[i] << " "; \
            } \
            std::cout << std::endl; \
        } while(0)

    static void tridiagonal_partial_pivoting_solver_dispatch(int                      m,
                                                             int                      n,
                                                             const float*             lower_diag,
                                                             const float*             main_diag,
                                                             const float*             upper_diag,
                                                             const float*             B,
                                                             float*                   X,
                                                             const tridiagonal_descr* descr)
    {
        constexpr int BLOCKSIZE = 256;
        constexpr int BLOCKDIM  = 4;

        std::cout << "m: " << m << " BLOCKDIM: " << BLOCKDIM << std::endl;

        std::vector<float> h_lower(m);
        std::vector<float> h_main(m);
        std::vector<float> h_upper(m);
        std::vector<float> h_B(m * n);
        CHECK_CUDA(
            cudaMemcpy(h_lower.data(), lower_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_main.data(), main_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(h_upper.data(), upper_diag, sizeof(float) * m, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_B.data(), B, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

        DEBUG_PRINT_ARRAY(h_main.data(), m, "h_main");
        DEBUG_PRINT_ARRAY(h_B.data(), m * n, "h_B");

        int m_pad = next_power_of_two(m);

        std::cout << "m: " << m << ", m_pad: " << m_pad << std::endl;

        data_marshaling_kernel<BLOCKSIZE, BLOCKDIM>
            <<<(m - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(m,
                                                     m_pad,
                                                     lower_diag,
                                                     main_diag,
                                                     upper_diag,
                                                     B,
                                                     descr->lower_pad,
                                                     descr->main_pad,
                                                     descr->upper_pad,
                                                     descr->B_pad);
        CHECK_CUDA_LAUNCH_ERROR();

        std::vector<float> h_lower_pad(m_pad);
        std::vector<float> h_main_pad(m_pad);
        std::vector<float> h_upper_pad(m_pad);
        std::vector<float> h_B_pad(m_pad * n);
        CHECK_CUDA(cudaMemcpy(
            h_lower_pad.data(), descr->lower_pad, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            h_main_pad.data(), descr->main_pad, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            h_upper_pad.data(), descr->upper_pad, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            h_B_pad.data(), descr->B_pad, sizeof(float) * m_pad * n, cudaMemcpyDeviceToHost));
        DEBUG_PRINT_ARRAY(h_lower_pad.data(), m_pad, "h_lower_pad");
        DEBUG_PRINT_ARRAY(h_main_pad.data(), m_pad, "h_main_pad");
        DEBUG_PRINT_ARRAY(h_upper_pad.data(), m_pad, "h_upper_pad");
        DEBUG_PRINT_ARRAY(h_B_pad.data(), m_pad * n, "h_B_pad");

        CHECK_CUDA(cudaMemset(descr->w_pad, 0, sizeof(float) * m_pad));
        CHECK_CUDA(cudaMemset(descr->v_pad, 0, sizeof(float) * m_pad));

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        std::cout << "Launching LBMT_solve_kernel with grid: " << grid << ", block: " << BLOCKSIZE
                  << std::endl;

        float* temp_a = nullptr;
        float* temp_b = nullptr;
        float* temp_c = nullptr;
        //CHECK_CUDA(cudaMalloc((void**)&temp_a, sizeof(float) * BLOCKDIM));
        //CHECK_CUDA(cudaMalloc((void**)&temp_b, sizeof(float) * BLOCKDIM));
        //CHECK_CUDA(cudaMalloc((void**)&temp_c, sizeof(float) * BLOCKDIM));
        //CHECK_CUDA(cudaMemset(temp_a, 0, sizeof(float) * BLOCKDIM));
        //CHECK_CUDA(cudaMemset(temp_b, 0, sizeof(float) * BLOCKDIM));
        //CHECK_CUDA(cudaMemset(temp_c, 0, sizeof(float) * BLOCKDIM));

        LBMT_solve_kernel<BLOCKSIZE, BLOCKDIM><<<grid, BLOCKSIZE>>>(m_pad,
                                                                    n,
                                                                    descr->lower_pad,
                                                                    descr->main_pad,
                                                                    descr->upper_pad,
                                                                    descr->w_pad,
                                                                    descr->v_pad,
                                                                    descr->mt,
                                                                    descr->B_pad,
                                                                    descr->pivot,
                                                                    temp_a,
                                                                    temp_b,
                                                                    temp_c);

        std::vector<float> h_w_pad(m_pad);
        std::vector<float> h_v_pad(m_pad);
        std::vector<float> h_mt_pad(m_pad);
        CHECK_CUDA(cudaMemcpy(
            h_w_pad.data(), descr->w_pad, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            h_v_pad.data(), descr->v_pad, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(h_mt_pad.data(), descr->mt, sizeof(float) * m_pad, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_B_pad.data(), descr->B_pad, sizeof(float) * m_pad * n, cudaMemcpyDeviceToHost));

        DEBUG_PRINT_ARRAY(h_w_pad.data(), m_pad, "h_w_pad");
        DEBUG_PRINT_ARRAY(h_v_pad.data(), m_pad, "h_v_pad");
        DEBUG_PRINT_ARRAY(h_mt_pad.data(), m_pad, "h_mt_pad");
        DEBUG_PRINT_ARRAY(h_B_pad.data(), m_pad * n, "h_B_pad");







        // Create tridiagonal S matrix
        int s_size = 2 * m_pad / BLOCKDIM;
        std::vector<float> h_S_lower(s_size);
        std::vector<float> h_S_main(s_size);
        std::vector<float> h_S_upper(s_size);
        std::vector<float> h_y(s_size * n);
        for(int i = 0; i < s_size; i++)
        {
            h_S_upper[i] = (i % 2 == 0) ? h_v_pad[i / 2] : 1.0f;
            h_S_lower[i] = (i % 2 == 0) ? 1.0f : h_w_pad[i / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }
        h_S_lower[0] = 0.0f;
        h_S_lower[1] = 0.0f;
        h_S_upper[s_size - 2] = 0.0f;
        h_S_upper[s_size - 1] = 0.0f;

        h_S_main[0] = 1.0f;
        h_S_main[s_size - 1] = 1.0f;
        for(int i = 1; i < s_size - 1; i++)
        {
            h_S_main[i] = (i % 2 == 0) ? h_w_pad[i / 2] : h_v_pad[i / 2 + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        for(int i = 0; i < s_size / 2; i++)
        {
            h_y[2 * i] = h_B_pad[i];
            h_y[2 * i + 1] = h_B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)];
        }

        std::vector<float> h_S_full(s_size * s_size, 0.0f);
        for(int i = 0; i < s_size; i++)
        {
            h_S_full[i * s_size + i] = h_S_main[i];
            if(i > 0)
                h_S_full[i * s_size + (i - 1)] = h_S_lower[i];
            if(i < s_size - 1)
                h_S_full[i * s_size + (i + 1)] = h_S_upper[i];
        }
        std::cout << "S matrix:" << std::endl;
        for(int i = 0; i < s_size; i++)
        {
            for(int j = 0; j < s_size; j++)
            {
                std::cout << h_S_full[i * s_size + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "" << std::endl;

        DEBUG_PRINT_ARRAY(h_S_lower.data(), s_size, "h_S_lower");
        DEBUG_PRINT_ARRAY(h_S_main.data(), s_size, "h_S_main");
        DEBUG_PRINT_ARRAY(h_S_upper.data(), s_size, "h_S_upper");
        DEBUG_PRINT_ARRAY(h_y.data(), s_size, "h_y");

        std::cout << "m: " << m << " m_pad: " << m_pad << " s_size: " << s_size << std::endl;

        // Solve Sx = y on host for debugging
        host_thomas_algorithm(s_size,
                              n,
                              h_S_lower.data(),
                              h_S_main.data(),
                              h_S_upper.data(),
                              h_y.data(),
                              h_y.data());

        DEBUG_PRINT_ARRAY(h_y.data(), s_size * n, "Thomas solution h_y");

        // Write y back to B_pad
        for(int i = 1; i < s_size - 1; i += 2)
        {
            float temp = h_y[i];
            h_y[i] = h_y[i + 1];
            h_y[i + 1] = temp;
        }
        DEBUG_PRINT_ARRAY(h_y.data(), s_size * n, "After correction h_y");
        for(int i = 0; i < s_size / 2; i++)
        {
            h_B_pad[i] = h_y[2 * i];
            h_B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)] = h_y[2 * i + 1];
        }

        DEBUG_PRINT_ARRAY(h_B_pad.data(), m_pad * n, "h_B_pad"); // 32 correct up to here

        // Complete Sx = B_pad
        for(int i = 0; i < m_pad / BLOCKDIM; i++)
        {
            float x1 = (i >= 1) ? h_B_pad[(m_pad / BLOCKDIM) * (BLOCKDIM - 1) + (i - 1)] : 0.0f;
            float x2 = (i < (BLOCKDIM - 1)) ? h_B_pad[i + 1] : 0.0f;

            for(int j = 1; j < BLOCKDIM - 1; j++)
            {
                h_B_pad[(m_pad / BLOCKDIM) * j + i] = h_B_pad[(m_pad / BLOCKDIM) * j + i]
                                                    - h_w_pad[(m_pad / BLOCKDIM) * j + i] * x1
                                                    - h_v_pad[(m_pad / BLOCKDIM) * j + i] * x2;
            }
        }

        DEBUG_PRINT_ARRAY(h_B_pad.data(), m_pad * n, "Before transform back h_B_pad");

        CHECK_CUDA(cudaMemcpy(descr->B_pad, h_B_pad.data(), sizeof(float) * m_pad * n, cudaMemcpyHostToDevice));

        data_marshaling_kernel2<BLOCKSIZE, BLOCKDIM>
            <<<(m - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(m,
                                                     m_pad,
                                                     descr->B_pad,
                                                     X);



        // std::vector<float> htemp_a(BLOCKDIM);
        // std::vector<float> htemp_b(BLOCKDIM);
        // std::vector<float> htemp_c(BLOCKDIM);
        // CHECK_CUDA(
        //     cudaMemcpy(htemp_a.data(), temp_a, sizeof(float) * BLOCKDIM, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(
        //     cudaMemcpy(htemp_b.data(), temp_b, sizeof(float) * BLOCKDIM, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(
        //     cudaMemcpy(htemp_c.data(), temp_c, sizeof(float) * BLOCKDIM, cudaMemcpyDeviceToHost));

        // DEBUG_PRINT_ARRAY(htemp_a.data(), BLOCKDIM, "htemp_a");
        // DEBUG_PRINT_ARRAY(htemp_b.data(), BLOCKDIM, "htemp_b");
        // DEBUG_PRINT_ARRAY(htemp_c.data(), BLOCKDIM, "htemp_c");

        // CHECK_CUDA(cudaFree(temp_a));
        // CHECK_CUDA(cudaFree(temp_b));
        // CHECK_CUDA(cudaFree(temp_c));

        CHECK_CUDA_LAUNCH_ERROR();
    }
}

void linalg::cuda_tridiagonal_solver(int                      m,
                                     int                      n,
                                     const float*             lower_diag,
                                     const float*             main_diag,
                                     const float*             upper_diag,
                                     const float*             B,
                                     float*                   X,
                                     const tridiagonal_descr* descr)
{
    switch(descr->pivoting_strategy)
    {
    case pivoting_strategy::none:
        tridiagonal_nonpivoting_solver_dispatch(
            m, n, lower_diag, main_diag, upper_diag, B, X, descr);
        break;
    case pivoting_strategy::partial:
        tridiagonal_partial_pivoting_solver_dispatch(
            m, n, lower_diag, main_diag, upper_diag, B, X, descr);
        break;
    }
}
