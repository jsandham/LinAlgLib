#include <Vector>
#include <cassert>
#include <iostream>
#include <map>

#include "../../../../include/direct_solvers/tridiagonal/tridiagonal.h"

#include "cuda_tridiagonal_spike_solver.h"

#include "tridiagonal_spike_kernels.cuh"

#include "../../../trace.h"

namespace linalg
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static void launch_data_marshaling_gemini(int      m,
                                              int      m_pad,
                                              const T* lower,
                                              const T* main,
                                              const T* upper,
                                              T*       lower_pad,
                                              T*       main_pad,
                                              T*       upper_pad)
    {
        ROUTINE_TRACE("launch_data_marshaling_gemini");

        const int nblocks = m_pad / BLOCKDIM;

        const int grid_size = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;

        data_marshaling_kernel_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>
            <<<grid_size, BLOCKSIZE>>>(
                m, m_pad, lower, main, upper, lower_pad, main_pad, upper_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static void launch_data_marshaling_B_gemini(int m, int m_pad, int n, const T* B, T* B_pad)
    {
        ROUTINE_TRACE("launch_data_marshaling_B_gemini");

        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        data_marshaling_B_kernel_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>
            <<<grid, block>>>(m, m_pad, n, B, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t PARTITIONS_PER_GROUP, typename T>
    static void
        launch_reverse_data_marshaling_B_gemini(int m, int m_pad, int n, const T* B_pad, T* B)
    {
        ROUTINE_TRACE("launch_reverse_data_marshaling_B_gemini");

        const int nblocks = m_pad / BLOCKDIM;

        const int grid_x = (nblocks + PARTITIONS_PER_GROUP - 1) / PARTITIONS_PER_GROUP;
        const int grid_y = std::min(n, 32768);
        const int grid_z = 1;

        const dim3 grid(grid_x, grid_y, grid_z);
        const dim3 block(BLOCKSIZE, 1, 1);

        reverse_data_marshaling_B_kernel_gemini<BLOCKSIZE, BLOCKDIM, PARTITIONS_PER_GROUP>
            <<<grid, block>>>(m, m_pad, n, B_pad, B);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_data_marshaling(int      m,
                                       int      m_pad,
                                       const T* lower,
                                       const T* main,
                                       const T* upper,
                                       T*       lower_pad,
                                       T*       main_pad,
                                       T*       upper_pad)
    {
        ROUTINE_TRACE("launch_data_marshaling");
        data_marshaling_kernel<BLOCKSIZE, BLOCKDIM><<<(m_pad - 1) / BLOCKSIZE + 1, BLOCKSIZE>>>(
            m, m_pad, lower, main, upper, lower_pad, main_pad, upper_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_data_marshaling_B(int m, int m_pad, int n, const T* B, T* B_pad)
    {
        ROUTINE_TRACE("launch_data_marshaling_B");
        data_marshaling_B_kernel<BLOCKSIZE, BLOCKDIM>
            <<<dim3((m_pad - 1) / BLOCKSIZE + 1, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m, m_pad, n, B, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve(int      m_pad,
                                  int      n,
                                  const T* lower,
                                  const T* main,
                                  const T* upper,
                                  T*       w_pad,
                                  T*       v_pad,
                                  T*       mt_pad,
                                  T*       B_pad)
    {
        ROUTINE_TRACE("launch_LBMT_solve");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        LBMT_solve_kernel<BLOCKSIZE, BLOCKDIM><<<dim3(grid, 1, 1), dim3(BLOCKSIZE, 1, 1)>>>(
            m_pad, 1, lower, main, upper, w_pad, v_pad, mt_pad, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve_wvmt(
        int m_pad, T* lower_pad, T* main_pad, T* upper_pad, T* w_pad, T* v_pad, T* mt_pad)
    {
        ROUTINE_TRACE("launch_LBMT_solve_wvmt");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        LBMT_solve_wvmt_kernel<BLOCKSIZE, BLOCKDIM><<<dim3(grid, 1, 1), dim3(BLOCKSIZE, 1, 1)>>>(
            m_pad, lower_pad, main_pad, upper_pad, w_pad, v_pad, mt_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve_rhs(int      m_pad,
                                      int      n,
                                      const T* lower_pad,
                                      const T* main_pad,
                                      const T* upper_pad,
                                      const T* mt_pad,
                                      T*       B_pad)
    {
        ROUTINE_TRACE("launch_LBMT_solve_rhs");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        LBMT_solve_rhs_kernel<BLOCKSIZE, BLOCKDIM>
            <<<dim3(grid, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m_pad, n, lower_pad, main_pad, upper_pad, mt_pad, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_fill_s_matrix(int      m_pad,
                                     int      n,
                                     const T* w,
                                     const T* v,
                                     const T* B_pad,
                                     T*       S_lower,
                                     T*       S_main,
                                     T*       S_upper,
                                     T*       S_B)
    {
        ROUTINE_TRACE("launch_fill_s_matrix");

        const int s_size = 2 * m_pad / BLOCKDIM;
        const int s_grid = (s_size - 1) / BLOCKSIZE + 1;
        fill_s_matrix_kernel<BLOCKSIZE, BLOCKDIM>
            <<<dim3(s_grid, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m_pad, n, w, v, B_pad, S_lower, S_main, S_upper, S_B);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t S_SIZE, typename T>
    static void launch_s_solve_kernel(int m,
                                      int n,
                                      const T* __restrict__ S_lower,
                                      const T* __restrict__ S_main,
                                      const T* __restrict__ S_upper,
                                      T* __restrict__ S_rhs)
    {
        ROUTINE_TRACE("launch_s_solve_kernel");
        S_solve_kernel<S_SIZE><<<n, 1>>>(m, n, S_lower, S_main, S_upper, S_rhs);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    // template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    // static void launch_s_solve_fused_kernel(int m,
    //                                         int n,
    //                                         const T* __restrict__ S_lower,
    //                                         const T* __restrict__ S_main,
    //                                         const T* __restrict__ S_upper,
    //                                         T* __restrict__ S_rhs)
    // {
    //     ROUTINE_TRACE("launch_s_solve_fused_kernel");

    //     S_solve_fused_kernel<BLOCKSIZE, BLOCKDIM>
    //         <<<n, BLOCKSIZE>>>(S_lower, S_main, S_upper, S_rhs);
    //     CHECK_CUDA_LAUNCH_ERROR();
    // }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_s_solve_fused_kernel(int m,
                                            int n,
                                            const T* __restrict__ S_lower,
                                            const T* __restrict__ S_main,
                                            const T* __restrict__ S_upper,
                                            T* __restrict__ S_rhs)
    {
        ROUTINE_TRACE("launch_s_solve_fused_kernel");

        S_solve_fused_kernel<BLOCKSIZE, BLOCKDIM, 1>
            <<<n, BLOCKSIZE>>>(m, S_lower, S_main, S_upper, S_rhs);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_scatter_S_B_to_B_pad(int m_pad, int n, const T* S_B, T* B_pad)
    {
        ROUTINE_TRACE("launch_scatter_S_B_to_B_pad");

        const int s_size = 2 * m_pad / BLOCKDIM;
        dim3      scatter_grid((s_size / 2 + BLOCKSIZE - 1) / BLOCKSIZE, std::min(n, 32768));
        scatter_S_B_to_B_pad_kernel<BLOCKSIZE, BLOCKDIM>
            <<<scatter_grid, BLOCKSIZE>>>(s_size, m_pad, n, S_B, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_backward_solve(int m_pad, int n, const T* w_pad, const T* v_pad, T* B_pad)
    {
        ROUTINE_TRACE("launch_backward_solve");

        const int grid = (m_pad - 1) / BLOCKSIZE + 1;
        backward_solve_kernel<BLOCKSIZE, BLOCKDIM>
            <<<dim3(grid, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m_pad, n, w_pad, v_pad, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_data_marshaling3(
        int m, int m_pad, int n, const T* w_pad, const T* v_pad, const T* B_pad, T* X)
    {
        ROUTINE_TRACE("launch_data_marshaling3");

        data_marshaling_kernel3<BLOCKSIZE, BLOCKDIM>
            <<<dim3((m_pad - 1) / BLOCKSIZE + 1, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m, m_pad, n, w_pad, v_pad, B_pad, X);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_data_marshaling2(int m, int m_pad, int n, const T* B_pad, T* X)
    {
        ROUTINE_TRACE("launch_data_marshaling2");

        data_marshaling_kernel2<BLOCKSIZE, BLOCKDIM>
            <<<dim3((m_pad - 1) / BLOCKSIZE + 1, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(
                m, m_pad, n, B_pad, X);
        CHECK_CUDA_LAUNCH_ERROR();
    }
}

template <typename T>
void linalg::tridiagonal_spike_solver(int      m,
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
                                      int      level)
{
    ROUTINE_TRACE("tridiagonal_spike_solver");

    constexpr int BLOCKDIM  = linalg::pivoting_data<T>::block_dim;
    constexpr int BLOCKSIZE = 256;

    const int m_pad = ((m + BLOCKDIM - 1) / BLOCKDIM) * BLOCKDIM;

    //std::cout << "m_pad: " << m_pad << " nblocks: " << (m_pad / BLOCKDIM) << std::endl;

    const int s_size = 2 * m_pad / BLOCKDIM;

    // launch_data_marshaling<BLOCKSIZE, BLOCKDIM>(m,
    //                                             m_pad,
    //                                             lower_diag,
    //                                             main_diag,
    //                                             upper_diag,
    //                                             lower_pad[level],
    //                                             main_pad[level],
    //                                             upper_pad[level]);
    // launch_data_marshaling_B<BLOCKSIZE, BLOCKDIM>(m, m_pad, n, B, B_pad[level]);

    launch_data_marshaling_gemini<128, BLOCKDIM, 256>(m,
                                                      m_pad,
                                                      lower_diag,
                                                      main_diag,
                                                      upper_diag,
                                                      lower_pad[level],
                                                      main_pad[level],
                                                      upper_pad[level]);
    launch_data_marshaling_B_gemini<128, BLOCKDIM, 256>(m, m_pad, n, B, B_pad[level]);

    CHECK_CUDA(cudaMemset(w_pad[level], 0, sizeof(T) * m_pad));
    CHECK_CUDA(cudaMemset(v_pad[level], 0, sizeof(T) * m_pad));

    if(n == 1)
    {
        launch_LBMT_solve<BLOCKSIZE, BLOCKDIM>(m_pad,
                                               n,
                                               lower_pad[level],
                                               main_pad[level],
                                               upper_pad[level],
                                               w_pad[level],
                                               v_pad[level],
                                               mt_pad[level],
                                               B_pad[level]);
    }
    else
    {
        launch_LBMT_solve_wvmt<BLOCKSIZE, BLOCKDIM>(m_pad,
                                                    lower_pad[level],
                                                    main_pad[level],
                                                    upper_pad[level],
                                                    w_pad[level],
                                                    v_pad[level],
                                                    mt_pad[level]);

        launch_LBMT_solve_rhs<BLOCKSIZE, BLOCKDIM>(m_pad,
                                                   n,
                                                   lower_pad[level],
                                                   main_pad[level],
                                                   upper_pad[level],
                                                   mt_pad[level],
                                                   B_pad[level]);
    }

    launch_fill_s_matrix<BLOCKSIZE, BLOCKDIM>(m_pad,
                                              n,
                                              w_pad[level],
                                              v_pad[level],
                                              B_pad[level],
                                              S_lower[level],
                                              S_main[level],
                                              S_upper[level],
                                              S_B[level]);

    using S_solve_launch_ptr = void (*)(int, int, const T*, const T*, const T*, T*);

    // static const std::map<int, S_solve_launch_ptr> s_solve_dispatch = {
    //     {2, launch_s_solve_kernel<2>},
    //     {4, launch_s_solve_kernel<4>},
    //     {6, launch_s_solve_kernel<6>},
    //     {8, launch_s_solve_kernel<8>},
    //     {10, launch_s_solve_kernel<10>},
    //     {12, launch_s_solve_kernel<12>},
    //     {14, launch_s_solve_kernel<14>},
    //     {16, launch_s_solve_kernel<16>},
    //     {18, launch_s_solve_kernel<18>},
    //     {20, launch_s_solve_kernel<20>},
    //     {22, launch_s_solve_kernel<22>},
    //     {24, launch_s_solve_kernel<24>},
    //     {26, launch_s_solve_kernel<26>},
    //     {28, launch_s_solve_kernel<28>},
    //     {30, launch_s_solve_kernel<30>},
    //     {32, launch_s_solve_kernel<32>},
    // };
    static const std::map<int, S_solve_launch_ptr> s_solve_dispatch = {
        {2, launch_s_solve_kernel<2>},
        {4, launch_s_solve_kernel<4>},
        {6, launch_s_solve_kernel<6>},
        {8, launch_s_solve_kernel<8>},
        {10, launch_s_solve_kernel<10>},
        {12, launch_s_solve_kernel<12>},
        {14, launch_s_solve_kernel<14>},
        {16, launch_s_solve_kernel<16>},
        {18, launch_s_solve_kernel<18>},
        {20, launch_s_solve_kernel<20>},
        {22, launch_s_solve_kernel<22>},
        {24, launch_s_solve_kernel<24>},
        {26, launch_s_solve_kernel<26>},
        {28, launch_s_solve_kernel<28>},
        {30, launch_s_solve_kernel<30>},
        {32, launch_s_solve_kernel<32>},
        {64, launch_s_solve_fused_kernel<32, 2>},
        {128, launch_s_solve_fused_kernel<32, 4>},
        {256, launch_s_solve_fused_kernel<32, 8>},
        {512, launch_s_solve_fused_kernel<32, 16>},
        {1024, launch_s_solve_fused_kernel<32, 32>},
    };

    auto dispatch_it = s_solve_dispatch.lower_bound(s_size);
    if(dispatch_it != s_solve_dispatch.end())
    {
        dispatch_it->second(s_size, n, S_lower[level], S_main[level], S_upper[level], S_B[level]);
    }
    else
    {
        if((level + 1) > pivoting_data<T>::tridiagonal_max_recursion_levels)
        {
            std::cout << "Error: Too many levels required " << level << std::endl;
            return;
        }

        tridiagonal_spike_solver(s_size,
                                 n,
                                 S_lower[level],
                                 S_main[level],
                                 S_upper[level],
                                 S_B[level],
                                 S_B[level], // reuse S_B as the rhs for the recursive call
                                 lower_pad,
                                 main_pad,
                                 upper_pad,
                                 B_pad,
                                 w_pad,
                                 v_pad,
                                 mt_pad,
                                 S_lower,
                                 S_main,
                                 S_upper,
                                 S_B,
                                 level + 1);
    }

    launch_scatter_S_B_to_B_pad<BLOCKSIZE, BLOCKDIM>(m_pad, n, S_B[level], B_pad[level]);

    launch_backward_solve<BLOCKSIZE, BLOCKDIM>(m_pad, n, w_pad[level], v_pad[level], B_pad[level]);

    launch_reverse_data_marshaling_B_gemini<128, BLOCKDIM, 256>(m, m_pad, n, B_pad[level], X);

    // launch_data_marshaling2<BLOCKSIZE, BLOCKDIM>(m, m_pad, n, B_pad[level], X);
    // launch_data_marshaling3<BLOCKSIZE, BLOCKDIM>(m, m_pad, n, w_pad[level], v_pad[level], B_pad[level], X);
}

// Explicit instantiations
template void linalg::tridiagonal_spike_solver<double>(int,
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
                                                       int);
template void linalg::tridiagonal_spike_solver<float>(int,
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
                                                      int);
