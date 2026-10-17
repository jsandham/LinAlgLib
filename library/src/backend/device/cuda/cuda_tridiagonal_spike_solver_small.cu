#include <Vector>
#include <cassert>
#include <iostream>
#include <map>

#include "../../../../include/direct_solvers/tridiagonal/tridiagonal.h"

#include "cuda_tridiagonal_spike_solver_small.h"

#include "tridiagonal_spike_fused_kernels.cuh"
#include "tridiagonal_spike_kernels.cuh"
#include "tridiagonal_spike_one_thread_per_system_kernels.cuh"

#include "../../../trace.h"

namespace linalg
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_transpose(int m, int n, const T* B, T* X)
    {
        ROUTINE_TRACE("launch_transpose");

        static_assert(BLOCKSIZE % BLOCKDIM == 0);
        static_assert(BLOCKSIZE >= BLOCKDIM);

        constexpr int TILE_Y = BLOCKSIZE / BLOCKDIM;
        constexpr int TILE_X = BLOCKDIM;

        const int gridx = (m + TILE_X - 1) / TILE_X;
        const int gridy = (n + TILE_Y - 1) / TILE_Y;

        dim3 threadsPerBlock(TILE_X, TILE_Y);
        dim3 numBlocks(gridx, std::min(gridy, 32768));

        const int ldb = m;
        const int ldx = n;

        transpose_kernel<TILE_X, TILE_Y><<<numBlocks, threadsPerBlock>>>(m, n, ldb, ldx, B, X);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_solve_one_thread_per_system(int m,
                                                   int n,
                                                   const T* __restrict__ lower,
                                                   const T* __restrict__ main,
                                                   const T* __restrict__ upper,
                                                   T* __restrict__ rhs)
    {
        ROUTINE_TRACE("launch_solve_one_thread_per_system");

        solve_one_thread_per_system_kernel<BLOCKSIZE, BLOCKDIM>
            <<<(n - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(m, n, lower, main, upper, rhs);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <typename T>
    static void launch_LBMT_solve_mt(int m,
                                     const T* __restrict__ lower,
                                     const T* __restrict__ main,
                                     const T* __restrict__ upper,
                                     T* __restrict__ mt)
    {
        ROUTINE_TRACE("launch_LBMT_solve_mt");

        LBMT_solve_mt_kernel<<<1, 1>>>(m, lower, main, upper, mt);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve_rhs_one_thread_per_system(int m,
                                                            int n,
                                                            const T* __restrict__ lower,
                                                            const T* __restrict__ main,
                                                            const T* __restrict__ upper,
                                                            const T* __restrict__ mt,
                                                            T* __restrict__ rhs)
    {
        ROUTINE_TRACE("launch_LBMT_solve_rhs_one_thread_per_system");

        LBMT_solve_rhs_one_thread_per_system_kernel<BLOCKSIZE, BLOCKDIM>
            <<<(n - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(m, n, lower, main, upper, mt, rhs);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    static void launch_solve_fused(int m,
                                   int n,
                                   const T* __restrict__ lower,
                                   const T* __restrict__ main,
                                   const T* __restrict__ upper,
                                   const T* __restrict__ B,
                                   T* __restrict__ X)
    {
        ROUTINE_TRACE("launch_solve_fused");

        solve_fused_kernel<BLOCKSIZE, BLOCKDIM, NCOLS>
            <<<n / NCOLS, BLOCKSIZE>>>(m, lower, main, upper, B, X);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_solve_fused_wvmt(int m,
                                        const T* __restrict__ lower,
                                        const T* __restrict__ main,
                                        const T* __restrict__ upper,
                                        T* __restrict__ coeff1,
                                        T* __restrict__ coeff2,
                                        T* __restrict__ coeff3,
                                        T* __restrict__ coeff4,
                                        T* __restrict__ coeff5,
                                        T* __restrict__ w,
                                        T* __restrict__ v,
                                        T* __restrict__ mt,
                                        int* __restrict__ pivot_mask)
    {
        ROUTINE_TRACE("launch_solve_fused_wvmt");

        solve_fused_wvmt_kernel<BLOCKSIZE, BLOCKDIM><<<1, BLOCKSIZE>>>(m,
                                                                       lower,
                                                                       main,
                                                                       upper,
                                                                       coeff1,
                                                                       coeff2,
                                                                       coeff3,
                                                                       coeff4,
                                                                       coeff5,
                                                                       w,
                                                                       v,
                                                                       mt,
                                                                       pivot_mask);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t NCOLS, typename T>
    static void launch_solve_fused2(int m,
                                    int n,
                                    const T* __restrict__ coeff1,
                                    const T* __restrict__ coeff2,
                                    const T* __restrict__ coeff3,
                                    const T* __restrict__ coeff4,
                                    const T* __restrict__ coeff5,
                                    const T* __restrict__ w,
                                    const T* __restrict__ v,
                                    const T* __restrict__ mt,
                                    const int* __restrict__ mask,
                                    const T* __restrict__ B,
                                    T* __restrict__ X)
    {
        ROUTINE_TRACE("launch_solve_fused2");

        solve_fused_kernel2<BLOCKSIZE, BLOCKDIM, NCOLS><<<n / NCOLS, BLOCKSIZE>>>(
            m, coeff1, coeff2, coeff3, coeff4, coeff5, w, v, mt, mask, B, X);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void tridiagonal_spike_solver_one_thread_per_system(int m,
                                                               int n,
                                                               const T* __restrict__ lower_diag,
                                                               const T* __restrict__ main_diag,
                                                               const T* __restrict__ upper_diag,
                                                               const T* __restrict__ B,
                                                               T* __restrict__ X,
                                                               T** B_pad,
                                                               T** mt,
                                                               int level)
    {
        ROUTINE_TRACE("tridiagonal_spike_solver_one_thread_per_system");

        // launch_transpose<BLOCKSIZE, BLOCKDIM>(m, n, B, B_pad[level]);
        // launch_solve_one_thread_per_system<BLOCKSIZE, BLOCKDIM>(
        //     m, n, lower_diag, main_diag, upper_diag, B_pad[level]);
        // launch_transpose<BLOCKSIZE, BLOCKDIM>(n, m, B_pad[level], X);

        launch_transpose<BLOCKSIZE, BLOCKDIM>(m, n, B, B_pad[level]);
        launch_LBMT_solve_mt(m, lower_diag, main_diag, upper_diag, mt[level]);
        launch_LBMT_solve_rhs_one_thread_per_system<BLOCKSIZE, BLOCKDIM>(
            m, n, lower_diag, main_diag, upper_diag, mt[level], B_pad[level]);
        launch_transpose<BLOCKSIZE, BLOCKDIM>(n, m, B_pad[level], X);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void tridiagonal_spike_solver_fused(int m,
                                               int n,
                                               const T* __restrict__ lower,
                                               const T* __restrict__ main,
                                               const T* __restrict__ upper,
                                               const T* __restrict__ B,
                                               T* __restrict__ X)
    {
        ROUTINE_TRACE("tridiagonal_spike_solver_fused");

        if(n < 500)
        {
            // For small number of columns we want to launch enough blocks to try and saturate the GPU
            // as much as possible. TODO: We should tune this based on the card being used.
            launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(m, n, lower, main, upper, B, X);
        }
        else
        {
            int remainder = n % 8;
            int nn        = n - remainder;

            launch_solve_fused<BLOCKSIZE, BLOCKDIM, 8>(m, nn, lower, main, upper, B, X);

            if(remainder > 0)
            {
                launch_solve_fused<BLOCKSIZE, BLOCKDIM, 1>(
                    m, remainder, lower, main, upper, B + m * nn, X + m * nn);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void tridiagonal_spike_solver_fused2(int m,
                                                int n,
                                                const T* __restrict__ lower,
                                                const T* __restrict__ main,
                                                const T* __restrict__ upper,
                                                const T* __restrict__ B,
                                                T* __restrict__ X,
                                                T* __restrict__ scratch,
                                                int* __restrict__ mask)
    {
        ROUTINE_TRACE("tridiagonal_spike_solver_fused2");

        T* coeff1 = scratch + 0 * BLOCKSIZE * BLOCKDIM;
        T* coeff2 = scratch + 1 * BLOCKSIZE * BLOCKDIM;
        T* coeff3 = scratch + 2 * BLOCKSIZE * BLOCKDIM;
        T* coeff4 = scratch + 3 * BLOCKSIZE * BLOCKDIM;
        T* coeff5 = scratch + 4 * BLOCKSIZE * BLOCKDIM;
        T* w      = scratch + 5 * BLOCKSIZE * BLOCKDIM;
        T* v      = scratch + 6 * BLOCKSIZE * BLOCKDIM;
        T* mt     = scratch + 7 * BLOCKSIZE * BLOCKDIM;

        CHECK_CUDA(cudaMemset(w, 0, sizeof(T) * BLOCKSIZE * BLOCKDIM));
        CHECK_CUDA(cudaMemset(v, 0, sizeof(T) * BLOCKSIZE * BLOCKDIM));
        // CHECK_CUDA(cudaMemset(scratch, 0, sizeof(T) * 8 * BLOCKSIZE * BLOCKDIM));

        launch_solve_fused_wvmt<BLOCKSIZE, BLOCKDIM>(
            m, lower, main, upper, coeff1, coeff2, coeff3, coeff4, coeff5, w, v, mt, mask);

        if(n < 500)
        {
            // For small number of columns we want to launch enough blocks to try and saturate the GPU
            // as much as possible. TODO: We should tune this based on the card being used.
            launch_solve_fused2<BLOCKSIZE, BLOCKDIM, 1>(
                m, n, coeff1, coeff2, coeff3, coeff4, coeff5, w, v, mt, mask, B, X);
        }
        else
        {
            int remainder = n % 8;
            int nn        = n - remainder;

            launch_solve_fused2<BLOCKSIZE, BLOCKDIM, 8>(
                m, nn, coeff1, coeff2, coeff3, coeff4, coeff5, w, v, mt, mask, B, X);

            if(remainder > 0)
            {
                launch_solve_fused2<BLOCKSIZE, BLOCKDIM, 1>(m,
                                                            remainder,
                                                            coeff1,
                                                            coeff2,
                                                            coeff3,
                                                            coeff4,
                                                            coeff5,
                                                            w,
                                                            v,
                                                            mt,
                                                            mask,
                                                            B + m * nn,
                                                            X + m * nn);
            }
        }
    }
}

template <typename T>
void linalg::tridiagonal_spike_solver_small(int      m,
                                            int      n,
                                            const T* lower_diag,
                                            const T* main_diag,
                                            const T* upper_diag,
                                            const T* B,
                                            T*       X,
                                            T**      lower_pad,
                                            T**      main_pad,
                                            T**      upper_pad,
                                            T**      B_pad,
                                            T**      w_pad,
                                            T**      v_pad,
                                            T**      mt_pad,
                                            T**      S_lower,
                                            T**      S_main,
                                            T**      S_upper,
                                            T**      S_B,
                                            T*       scratch,
                                            int*     mask,
                                            int      level)
{
    ROUTINE_TRACE("tridiagonal_spike_solver_small");

    using solve_launch_ptr
        = void (*)(int, int, const T*, const T*, const T*, const T*, T*, T**, T**, int);
    using solve_fused_launch_ptr = void (*)(int, int, const T*, const T*, const T*, const T*, T*);
    using solve_fused_launch_ptr2
        = void (*)(int, int, const T*, const T*, const T*, const T*, T*, T*, int*);

    static const std::map<int, solve_launch_ptr> solve_dispatch = {
        {4, tridiagonal_spike_solver_one_thread_per_system<256, 4>},
        {8, tridiagonal_spike_solver_one_thread_per_system<256, 8>},
        {16, tridiagonal_spike_solver_one_thread_per_system<256, 16>},
        {32, tridiagonal_spike_solver_one_thread_per_system<256, 32>},
        {64, tridiagonal_spike_solver_one_thread_per_system<256, 64>},
    };

    static const std::map<int, solve_fused_launch_ptr> solve_fused_dispatch = {
        {128, tridiagonal_spike_solver_fused<32, 4>},
        {256, tridiagonal_spike_solver_fused<32, 8>},
        {512, tridiagonal_spike_solver_fused<32, 16>},
        {1024, tridiagonal_spike_solver_fused<32, 32>},
    };

    static const std::map<int, solve_fused_launch_ptr2> solve_fused_dispatch2 = {
        {128, tridiagonal_spike_solver_fused2<32, 4>},
        {256, tridiagonal_spike_solver_fused2<32, 8>},
        {512, tridiagonal_spike_solver_fused2<32, 16>},
        {1024, tridiagonal_spike_solver_fused2<32, 32>},
    };

    if(m <= 64)
    {
        auto dispatch_it = solve_dispatch.lower_bound(m);
        if(dispatch_it != solve_dispatch.end())
        {
            dispatch_it->second(
                m, n, lower_diag, main_diag, upper_diag, B, X, B_pad, mt_pad, level);
        }
        else
        {
            std::cout << "Error: Dispath for problem size <= 64 failed." << std::endl;
            return;
        }
    }
    else
    {
        // auto dispatch_it = solve_fused_dispatch.lower_bound(m);
        // if(dispatch_it != solve_fused_dispatch.end())
        // {
        //     dispatch_it->second(m, n, lower_diag, main_diag, upper_diag, B, X);
        // }
        // else
        // {
        //     std::cout << "Error: Dispath for problem size <= 1024 failed." << std::endl;
        //     return;
        // }
        auto dispatch_it = solve_fused_dispatch2.lower_bound(m);
        if(dispatch_it != solve_fused_dispatch2.end())
        {
            dispatch_it->second(m, n, lower_diag, main_diag, upper_diag, B, X, scratch, mask);
        }
        else
        {
            std::cout << "Error: Dispath for problem size <= 1024 failed." << std::endl;
            return;
        }
    }
}

// Explicit instantiations
template void linalg::tridiagonal_spike_solver_small<double>(int,
                                                             int,
                                                             const double*,
                                                             const double*,
                                                             const double*,
                                                             const double*,
                                                             double*,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double**,
                                                             double*,
                                                             int*,
                                                             int);
template void linalg::tridiagonal_spike_solver_small<float>(int,
                                                            int,
                                                            const float*,
                                                            const float*,
                                                            const float*,
                                                            const float*,
                                                            float*,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float**,
                                                            float*,
                                                            int*,
                                                            int);
