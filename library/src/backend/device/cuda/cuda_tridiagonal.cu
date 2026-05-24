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
static constexpr int BLOCKDIM = 256;

struct linalg::tridiagonal_descr
{
    pivoting_strategy pivoting_strategy;

    // Buffers for non-pivoting approach (one per recursion level)
    double* lower_modified[MAX_RECURSION_LEVELS];
    double* main_modified[MAX_RECURSION_LEVELS];
    double* upper_modified[MAX_RECURSION_LEVELS];
    double* B_modified[MAX_RECURSION_LEVELS];

    double* spike_lower[MAX_RECURSION_LEVELS];
    double* spike_main[MAX_RECURSION_LEVELS];
    double* spike_upper[MAX_RECURSION_LEVELS];
    double* spike_B[MAX_RECURSION_LEVELS];
    double* spike_X[MAX_RECURSION_LEVELS];

    // Buffers for partial pivoting approach (to be implemented)
    double* lower_pad;
    double* main_pad;
    double* upper_pad;
    double* B_pad;

    double* w_pad;
    double* v_pad;

    double* mt;

    double* S_lower;
    double* S_main;
    double* S_upper;
    double* S_B;
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
    if(descr->mt != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->mt));
        descr->mt = nullptr;
    }
    if(descr->S_lower != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->S_lower));
        descr->S_lower = nullptr;
    }
    if(descr->S_main != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->S_main));
        descr->S_main = nullptr;
    }
    if(descr->S_upper != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->S_upper));
        descr->S_upper = nullptr;
    }
    if(descr->S_B != nullptr)
    {
        CHECK_CUDA(cudaFree(descr->S_B));
        descr->S_B = nullptr;
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
                                                          const double*      lower_diag,
                                                          const double*      main_diag,
                                                          const double*      upper_diag,
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
                cudaMalloc((void**)&descr->lower_modified[level], sizeof(double) * current_m));
            CHECK_CUDA(
                cudaMalloc((void**)&descr->main_modified[level], sizeof(double) * current_m));
            CHECK_CUDA(
                cudaMalloc((void**)&descr->upper_modified[level], sizeof(double) * current_m));
            CHECK_CUDA(
                cudaMalloc((void**)&descr->B_modified[level], sizeof(double) * current_m * n));

            CHECK_CUDA(cudaMalloc((void**)&descr->spike_lower[level], sizeof(double) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_main[level], sizeof(double) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_upper[level], sizeof(double) * num_spikes));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_B[level], sizeof(double) * num_spikes * n));
            CHECK_CUDA(cudaMalloc((void**)&descr->spike_X[level], sizeof(double) * num_spikes * n));

            current_m = num_spikes;
        }
    }

    static void tridiagonal_partial_pivoting_analysis_dispatch(int                m,
                                                               int                n,
                                                               const double*      lower_diag,
                                                               const double*      main_diag,
                                                               const double*      upper_diag,
                                                               tridiagonal_descr* descr)
    {
        const int m_pad = next_power_of_two(m);

        CHECK_CUDA(cudaMalloc((void**)&descr->lower_pad, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->main_pad, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->upper_pad, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->B_pad, sizeof(double) * m_pad * n));

        CHECK_CUDA(cudaMalloc((void**)&descr->w_pad, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->v_pad, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMalloc((void**)&descr->mt, sizeof(double) * m_pad));

        const int S_size = 2 * m_pad / BLOCKDIM;

        CHECK_CUDA(cudaMalloc((void**)&descr->S_lower, sizeof(double) * S_size));
        CHECK_CUDA(cudaMalloc((void**)&descr->S_main, sizeof(double) * S_size));
        CHECK_CUDA(cudaMalloc((void**)&descr->S_upper, sizeof(double) * S_size));
        CHECK_CUDA(cudaMalloc((void**)&descr->S_B, sizeof(double) * S_size * n));
    }

    static void tridiagonal_nonpivoting_solver_dispatch(int                      m,
                                                        int                      n,
                                                        const double*            lower_diag,
                                                        const double*            main_diag,
                                                        const double*            upper_diag,
                                                        const double*            B,
                                                        double*                  X,
                                                        const tridiagonal_descr* descr,
                                                        int                      level = 0);
}

void linalg::cuda_tridiagonal_analysis(int                m,
                                       int                n,
                                       const double*      lower_diag,
                                       const double*      main_diag,
                                       const double*      upper_diag,
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
        constexpr int BLOCKSIZE = 256; // remember to change in analysis as well!
        constexpr int NUM_RHS
            = 1; //8; // dont forget to change this back when using float data type
        int nblocks    = ((m - 1) / BLOCKSIZE + 1);
        int num_spikes = 2 * nblocks;

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
        // crpcr_pow2_shared_multi_rhs_kernel<512, 256, 8>
        //     <<<((n - 1) / 8 + 1), 512>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
        crpcr_pow2_shared_multi_rhs_kernel<512, 256, 1>
            <<<((n - 1) / 1 + 1), 512>>>(m, n, lower_diag, main_diag, upper_diag, B, X);
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
                                                        const double*            lower_diag,
                                                        const double*            main_diag,
                                                        const double*            upper_diag,
                                                        const double*            B,
                                                        double*                  X,
                                                        const tridiagonal_descr* descr,
                                                        int                      level)
    {
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

    static void host_thomas_algorithm(int           m,
                                      int           n,
                                      const double* lower_diag,
                                      const double* main_diag,
                                      const double* upper_diag,
                                      double*       y)
    {
        std::vector<double> mt(m);
        std::vector<int> pivot_mask(m);

        int k = 0;
        double bk = main_diag[k];

        while(k < m)
        {
            double ck   = upper_diag[k];
            double ck_1 = (k < (m - 1)) ? upper_diag[k + 1] : static_cast<double>(0);
            double bk_1 = (k < (m - 1)) ? main_diag[k + 1] : static_cast<double>(0);
            double ak_1 = (k < (m - 1)) ? lower_diag[k + 1] : static_cast<double>(0);
            double ak_2 = (k < (m - 2)) ? lower_diag[k + 2] : static_cast<double>(0);

            // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
            // pivoting criteria
            const bool use_1x1_pivot = bunch_kaufman_criterion(ak_1, ak_2, bk, bk_1, ck, ck_1);

            // 1x1 pivoting
            if(use_1x1_pivot || k == (m - 1))
            {
                const double inv_bk = static_cast<double>(1) / bk;

                mt[k] = ck * inv_bk;

                pivot_mask[k] = 1; // mark this pivot as 1x1

                // L * B * x = y
                double rhsk = y[k] * inv_bk;

                y[k] = rhsk;

                if(k < (m - 1))
                {
                    y[k + 1] += -(ak_1 * rhsk);

                    bk_1 = bk_1 - ak_1 * ck * inv_bk;
                }

                bk = bk_1;

                k += 1;
            }
            else
            {
                const double det = static_cast<double>(1) / (bk * bk_1 - ak_1 * ck);

                mt[k] = -ck * ck_1 * det;

                pivot_mask[k] = 2;

                if(k < (m - 1))
                {
                    mt[k + 1] = bk * ck_1 * det;

                    pivot_mask[k + 1] = 2;
                }

                double bk_2 = static_cast<double>(0);

                // L * B * x = y
                double rhsk   = y[k] * det;
                double rhsk_1 = y[k + 1] * det;

                y[k]       = (bk_1 * rhsk - ck * rhsk_1);
                y[k + 1] = (-ak_1 * rhsk + bk * rhsk_1);

                if(k < (m - 2))
                {
                    y[k + 2] += -(-ak_1 * ak_2 * rhsk + ak_2 * bk * rhsk_1);

                    bk_2 = main_diag[k + 2];
                    bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
                }

                bk = bk_2;
                k += 2;
            }
        }

        assert(k == m);
        // at this point k = m. Could just set k = m - 1 here
        k--;

        k -= pivot_mask[k];

        // backward solve (M^T * y = y)
        while(k >= 0)
        {
            if(pivot_mask[k] == 1)
            {
                const double tmp = mt[k];

                y[k] += -tmp * y[k + 1];

                k -= 1;
            }
            else
            {
                const double tmp1 = mt[k];
                const double tmp2 = mt[k - 1];

                y[k] += -tmp1 * y[k + 1];
                y[k - 1] += -tmp2 * y[k + 1];

                k -= 2;
            }
        }
    }

#define DEBUG_PRINT_ARRAY(arr, size, name) \
    do                                     \
    {                                      \
        std::cout << name << ": ";         \
        for(int i = 0; i < size; i++)      \
        {                                  \
            std::cout << arr[i] << " ";    \
        }                                  \
        std::cout << std::endl;            \
    } while(0)

#define DEBUG_PRINT_TRIDIAG_MATRIX_VECTOR_PRODUCT(                                   \
    lower_pad, main_pad, upper_pad, B_pad, m_pad, blockdim, name)                    \
    do                                                                               \
    {                                                                                \
        for(int j = 0; j < (m_pad) / (blockdim); j++)                                \
        {                                                                            \
            std::vector<double> h_temp((blockdim), 0.0);                             \
            for(int i = 0; i < (blockdim); i++)                                      \
            {                                                                        \
                if(i == 0)                                                           \
                {                                                                    \
                    h_temp[i] = (main_pad)[(m_pad) / (blockdim) * i + j]             \
                                    * (B_pad)[(m_pad) / (blockdim) * i + j]          \
                                + (upper_pad)[(m_pad) / (blockdim) * i + j]          \
                                      * (B_pad)[(m_pad) / (blockdim) * (i + 1) + j]; \
                }                                                                    \
                else if(i == (blockdim) - 1)                                         \
                {                                                                    \
                    h_temp[i] = (lower_pad)[(m_pad) / (blockdim) * i + j]            \
                                    * (B_pad)[(m_pad) / (blockdim) * (i - 1) + j]    \
                                + (main_pad)[(m_pad) / (blockdim) * i + j]           \
                                      * (B_pad)[(m_pad) / (blockdim) * i + j];       \
                }                                                                    \
                else                                                                 \
                {                                                                    \
                    h_temp[i] = (lower_pad)[(m_pad) / (blockdim) * i + j]            \
                                    * (B_pad)[(m_pad) / (blockdim) * (i - 1) + j]    \
                                + (main_pad)[(m_pad) / (blockdim) * i + j]           \
                                      * (B_pad)[(m_pad) / (blockdim) * i + j]        \
                                + (upper_pad)[(m_pad) / (blockdim) * i + j]          \
                                      * (B_pad)[(m_pad) / (blockdim) * (i + 1) + j]; \
                }                                                                    \
            }                                                                        \
            DEBUG_PRINT_ARRAY(h_temp.data(), (blockdim), (name));                    \
        }                                                                            \
    } while(0)

#define DEBUG_PRINT_TRIDIAG_MATRIX(lower_diag, main_diag, upper_diag, matrix_size, name) \
    do                                                                                   \
    {                                                                                    \
        std::vector<double> h_A((matrix_size) * (matrix_size), 0.0);                     \
        for(int i = 0; i < (matrix_size); i++)                                           \
        {                                                                                \
            h_A[i * (matrix_size) + i] = (main_diag)[i];                                 \
            if(i > 0)                                                                    \
            {                                                                            \
                h_A[i * (matrix_size) + (i - 1)] = (lower_diag)[i];                      \
            }                                                                            \
            if(i < (matrix_size) - 1)                                                    \
            {                                                                            \
                h_A[i * (matrix_size) + (i + 1)] = (upper_diag)[i];                      \
            }                                                                            \
        }                                                                                \
        std::cout << (name) << std::endl;                                                \
        for(int r = 0; r < (matrix_size); r++)                                           \
        {                                                                                \
            for(int c = 0; c < (matrix_size); c++)                                       \
            {                                                                            \
                std::cout << h_A[r * (matrix_size) + c] << " ";                          \
            }                                                                            \
            std::cout << std::endl;                                                      \
        }                                                                                \
    } while(0)

    template <typename T, uint32_t S_SIZE>
    static void launch_s_solve_kernel(int n,
                                      const T* __restrict__ S_lower,
                                      const T* __restrict__ S_main,
                                      const T* __restrict__ S_upper,
                                      T* __restrict__ rhs)
    {
        S_solve_kernel<S_SIZE><<<1, 1>>>(n, S_lower, S_main, S_upper, rhs);
    }

    static void tridiagonal_partial_pivoting_solver_dispatch(int                      m,
                                                             int                      n,
                                                             const double*            lower_diag,
                                                             const double*            main_diag,
                                                             const double*            upper_diag,
                                                             const double*            B,
                                                             double*                  X,
                                                             const tridiagonal_descr* descr)
    {
        constexpr int BLOCKSIZE = 256;

        const int m_pad = next_power_of_two(m);

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

        CHECK_CUDA(cudaMemset(descr->w_pad, 0, sizeof(double) * m_pad));
        CHECK_CUDA(cudaMemset(descr->v_pad, 0, sizeof(double) * m_pad));

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        // LBM^T solve
        LBMT_solve_kernel<BLOCKSIZE, BLOCKDIM><<<grid, BLOCKSIZE>>>(m_pad,
                                                                    n,
                                                                    descr->lower_pad,
                                                                    descr->main_pad,
                                                                    descr->upper_pad,
                                                                    descr->w_pad,
                                                                    descr->v_pad,
                                                                    descr->mt,
                                                                    descr->B_pad);
        CHECK_CUDA_LAUNCH_ERROR();

        const int s_grid = ((2 * m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        // Create tridiagonal S matrix
        fill_s_matrix_kernel<BLOCKSIZE, BLOCKDIM><<<s_grid, BLOCKSIZE>>>(m_pad,
                                                                         n,
                                                                         descr->w_pad,
                                                                         descr->v_pad,
                                                                         descr->B_pad,
                                                                         descr->S_lower,
                                                                         descr->S_main,
                                                                         descr->S_upper,
                                                                         descr->S_B);
        CHECK_CUDA_LAUNCH_ERROR();

        int                 s_size = 2 * m_pad / BLOCKDIM;

        std::cout << "s_size: " << s_size << std::endl;

        using S_solve_launch_ptr
            = void (*)(int, const double*, const double*, const double*, double*);

        static const std::map<int, S_solve_launch_ptr> s_solve_dispatch = {
            {2, launch_s_solve_kernel<double, 2>},
            {4, launch_s_solve_kernel<double, 4>},
            {8, launch_s_solve_kernel<double, 8>},
            {16, launch_s_solve_kernel<double, 16>},
            {32, launch_s_solve_kernel<double, 32>},
            {64, launch_s_solve_kernel<double, 64>},
            {128, launch_s_solve_kernel<double, 128>},
            {256, launch_s_solve_kernel<double, 256>},
            {512, launch_s_solve_kernel<double, 512>},
            {1024, launch_s_solve_kernel<double, 1024>},
        };

        // auto dispatch_it = s_solve_dispatch.lower_bound(s_size);
        // if(dispatch_it != s_solve_dispatch.end())
        // {
        //    dispatch_it->second(n, descr->S_lower, descr->S_main, descr->S_upper, descr->S_B);
        // }
        // std::vector<double> h_y(s_size * n);
        // CHECK_CUDA(cudaMemcpy(
        //    h_y.data(), descr->S_B, sizeof(double) * s_size * n, cudaMemcpyDeviceToHost));

        // std::vector<double> h_B_pad(m_pad * n);
        // CHECK_CUDA(cudaMemcpy(
        //    h_B_pad.data(), descr->B_pad, sizeof(double) * m_pad * n, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(
        //   h_y.data(), descr->S_B, sizeof(double) * s_size * n, cudaMemcpyDeviceToHost));

        // Solve Sx = y on host for debugging
        std::vector<double> h_S_lower(s_size);
        std::vector<double> h_S_main(s_size);
        std::vector<double> h_S_upper(s_size);
        std::vector<double> h_y(s_size * n);
        CHECK_CUDA(cudaMemcpy(
           h_S_lower.data(), descr->S_lower, sizeof(double) * s_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
           h_S_main.data(), descr->S_main, sizeof(double) * s_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
           h_S_upper.data(), descr->S_upper, sizeof(double) * s_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            h_y.data(), descr->S_B, sizeof(double) * s_size * n, cudaMemcpyDeviceToHost));

        std::vector<double> h_B_pad(m_pad * n);
        CHECK_CUDA(cudaMemcpy(
            h_B_pad.data(), descr->B_pad, sizeof(double) * m_pad * n, cudaMemcpyDeviceToHost));
        {
            host_thomas_algorithm(s_size,
                                n,
                                h_S_lower.data(),
                                h_S_main.data(),
                                h_S_upper.data(),
                                h_y.data());
        }
        //DEBUG_PRINT_ARRAY(h_y.data(), s_size * n, "Thomas solution h_y");









        // Write y back to B_pad
        for(int i = 1; i < s_size - 1; i += 2)
        {
            double temp = h_y[i];
            h_y[i]      = h_y[i + 1];
            h_y[i + 1]  = temp;
        }
        //DEBUG_PRINT_ARRAY(h_y.data(), s_size * n, "After correction h_y");
        for(int i = 0; i < s_size / 2; i++)
        {
            h_B_pad[i]                                       = h_y[2 * i];
            h_B_pad[i + (m_pad / BLOCKDIM) * (BLOCKDIM - 1)] = h_y[2 * i + 1];
        }

        //DEBUG_PRINT_ARRAY(h_B_pad.data(), m_pad * n, "h_B_pad"); // 32 correct up to here

        CHECK_CUDA(cudaMemcpy(
            descr->B_pad, h_B_pad.data(), sizeof(double) * m_pad * n, cudaMemcpyHostToDevice));

        // Complete Sx = B_pad
        backward_solve_kernel<BLOCKSIZE, BLOCKDIM>
            <<<grid, BLOCKSIZE>>>(m_pad, n, descr->w_pad, descr->v_pad, descr->B_pad);
        CHECK_CUDA_LAUNCH_ERROR();

        data_marshaling_kernel2<BLOCKSIZE, BLOCKDIM>
            <<<(m - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(m, m_pad, descr->B_pad, X);
        CHECK_CUDA_LAUNCH_ERROR();
    }
}

void linalg::cuda_tridiagonal_solver(int                      m,
                                     int                      n,
                                     const double*            lower_diag,
                                     const double*            main_diag,
                                     const double*            upper_diag,
                                     const double*            B,
                                     double*                  X,
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
