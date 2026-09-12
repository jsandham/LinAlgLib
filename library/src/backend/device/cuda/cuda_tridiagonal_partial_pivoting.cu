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
#include <cassert>
#include <iostream>
#include <map>

#include "linalg_enums.h"

#include "../../../../include/direct_solvers/tridiagonal/tridiagonal.h"

#include "cuda_tridiagonal_partial_pivoting.h"

#include "tridiagonal_spike_kernels.cuh"

#include "../../../trace.h"

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
            std::vector<T> h_temp((blockdim), 0.0);                                  \
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
        std::vector<T> h_A((matrix_size) * (matrix_size), 0.0);                          \
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
                                  T*       w,
                                  T*       v,
                                  T*       mt,
                                  T*       B_pad)
    {
        ROUTINE_TRACE("launch_LBMT_solve");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        LBMT_solve_kernel<BLOCKSIZE, BLOCKDIM><<<dim3(grid, 1, 1), dim3(BLOCKSIZE, 1, 1)>>>(
            m_pad, 1, lower, main, upper, w, v, mt, B_pad);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve_wvmt(int           m_pad,
                                       T*      lower,
                                       T*      main,
                                       T*      upper,
                                       T*            w,
                                       T*            v,
                                       T*            mt)
    {
        ROUTINE_TRACE("launch_LBMT_solve_wvmt");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        LBMT_solve_wvmt_kernel<BLOCKSIZE, BLOCKDIM><<<dim3(grid, 1, 1), dim3(BLOCKSIZE, 1, 1)>>>(
            m_pad,
            lower,
            main,
            upper,
            w,
            v,
            mt);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    static void launch_LBMT_solve_rhs(int           m_pad,
                                      int           n,
                                      const T*      lower,
                                      const T*      main,
                                      const T*      upper,
                                      const T*      mt,
                                      T*            rhs)
    {
        ROUTINE_TRACE("launch_LBMT_solve_rhs");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;

        // std::cout << "grid: " << grid << std::endl;

        LBMT_solve_rhs_kernel<BLOCKSIZE, BLOCKDIM><<<dim3(grid, n, 1), dim3(BLOCKSIZE, 1, 1)>>>(
            m_pad,
            n,
            lower,
            main,
            upper,
            mt,
            rhs);
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

    template <typename T, uint32_t S_SIZE>
    static void launch_s_solve_kernel(int m,
                                      int n,
                                      const T* __restrict__ S_lower,
                                      const T* __restrict__ S_main,
                                      const T* __restrict__ S_upper,
                                      T* __restrict__ rhs)
    {
        ROUTINE_TRACE("launch_s_solve_kernel");
        S_solve_kernel<S_SIZE><<<n, 1>>>(m, n, S_lower, S_main, S_upper, rhs);
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
    static void launch_backward_solve(int m_pad, int n, const T* w, const T* v, T* rhs)
    {
        ROUTINE_TRACE("launch_backward_solve");

        const int grid = ((m_pad / BLOCKDIM) - 1) / BLOCKSIZE + 1;
        backward_solve_kernel<BLOCKSIZE, BLOCKDIM>
            <<<dim3(grid, std::min(n, 32768), 1), dim3(BLOCKSIZE, 1, 1)>>>(m_pad, n, w, v, rhs);
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

    template <typename T>
    static void tridiagonal_partial_pivoting_solver_dispatch(int            m,
                                                             int            n,
                                                             const T*       lower_diag,
                                                             const T*       main_diag,
                                                             const T*       upper_diag,
                                                             const T*       B,
                                                             T*             X,
                                                             T**            lower_pad,
                                                             T**            main_pad,
                                                             T**            upper_pad,
                                                             T**            B_pad,
                                                             T**            w_pad,
                                                             T**            v_pad,
                                                             T**            mt,
                                                             T**            S_lower,
                                                             T**            S_main,
                                                             T**            S_upper,
                                                             T**            S_B,
                                                             int            level = 0)
    {
        constexpr int BLOCKDIM  = linalg::pivoting_data<T>::block_dim;
        constexpr int BLOCKSIZE = 256;

        int m_pad = next_power_of_two(m);
        m_pad     = std::max(m_pad, BLOCKDIM);

        const int nblocks = m_pad / BLOCKDIM;
        const int s_size = 2 * m_pad / BLOCKDIM;

        launch_data_marshaling<BLOCKSIZE, BLOCKDIM>(m,
                                                    m_pad,
                                                    lower_diag,
                                                    main_diag,
                                                    upper_diag,
                                                    lower_pad[level],
                                                    main_pad[level],
                                                    upper_pad[level]);

        launch_data_marshaling_B<BLOCKSIZE, BLOCKDIM>(m, m_pad, n, B, B_pad[level]);

        // for(int batch = 0; batch < n; batch++)
        // {
        //     CHECK_CUDA(cudaMemset(w_pad[level], 0, sizeof(double) * m_pad));
        //     CHECK_CUDA(cudaMemset(v_pad[level], 0, sizeof(double) * m_pad));

        //     launch_LBMT_solve<BLOCKSIZE, BLOCKDIM>(m_pad,
        //                                            n,
        //                                            lower_pad[level],
        //                                            main_pad[level],
        //                                            upper_pad[level],
        //                                            w_pad[level],
        //                                            v_pad[level],
        //                                            mt[level],
        //                                            B_pad[level] + batch * m_pad);
        // }

        CHECK_CUDA(cudaMemset(w_pad[level], 0, sizeof(T) * m_pad));
        CHECK_CUDA(cudaMemset(v_pad[level], 0, sizeof(T) * m_pad));

        launch_LBMT_solve_wvmt<BLOCKSIZE, BLOCKDIM>(m_pad,
                                                    lower_pad[level],
                                                    main_pad[level],
                                                    upper_pad[level],
                                                    w_pad[level],
                                                    v_pad[level],
                                                    mt[level]);

        for(int i = 0; i < n; i += 32768)
        {
            launch_LBMT_solve_rhs<BLOCKSIZE, BLOCKDIM>(m_pad,
                                                       std::min(n - i, 32768),
                                                       lower_pad[level],
                                                       main_pad[level],
                                                       upper_pad[level],
                                                       mt[level],
                                                       B_pad[level] + m_pad * i);
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

        static const std::map<int, S_solve_launch_ptr> s_solve_dispatch = {
            {2, launch_s_solve_kernel<T, 2>},
            {4, launch_s_solve_kernel<T, 4>},
            {8, launch_s_solve_kernel<T, 8>},
            {16, launch_s_solve_kernel<T, 16>},
            {32, launch_s_solve_kernel<T, 32>},
            {64, launch_s_solve_kernel<T, 64>},
            {128, launch_s_solve_kernel<T, 128>},
            {256, launch_s_solve_kernel<T, 256>},
            {512, launch_s_solve_kernel<T, 512>},
            {1024, launch_s_solve_kernel<T, 1024>},
        };

        auto dispatch_it = s_solve_dispatch.lower_bound(s_size);
        if(dispatch_it != s_solve_dispatch.end())
        {
            dispatch_it->second(
                s_size, n, S_lower[level], S_main[level], S_upper[level], S_B[level]);
        }
        else
        {
            tridiagonal_partial_pivoting_solver_dispatch(
                s_size,
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
                mt,
                S_lower,
                S_main,
                S_upper,
                S_B,
                level + 1);
        }

        launch_scatter_S_B_to_B_pad<BLOCKSIZE, BLOCKDIM>(m_pad, n, S_B[level], B_pad[level]);

        launch_backward_solve<BLOCKSIZE, BLOCKDIM>(
            m_pad, n, w_pad[level], v_pad[level], B_pad[level]);

        launch_data_marshaling2<BLOCKSIZE, BLOCKDIM>(m, m_pad, n, B_pad[level], X);
    }
}

template <typename T>
void linalg::cuda_partial_pivoting_solver(int            m,
                                          int            n,
                                          const T*       lower_diag,
                                          const T*       main_diag,
                                          const T*       upper_diag,
                                          const T*       B,
                                          T*             X,
                                          T**            lower_pad,
                                          T**            main_pad,
                                          T**            upper_pad,
                                          T**            B_pad,
                                          T**            w_pad,
                                          T**            v_pad,
                                          T**            mt,
                                          T**            S_lower,
                                          T**            S_main,
                                          T**            S_upper,
                                          T**            S_B)
{
    tridiagonal_partial_pivoting_solver_dispatch(m,
                                                 n,
                                                 lower_diag,
                                                 main_diag,
                                                 upper_diag,
                                                 B,
                                                 X,
                                                 lower_pad,
                                                 main_pad,
                                                 upper_pad,
                                                 B_pad,
                                                 w_pad,
                                                 v_pad,
                                                 mt,
                                                 S_lower,
                                                 S_main,
                                                 S_upper,
                                                 S_B);
}

// Explicit instantiations
template void linalg::cuda_partial_pivoting_solver<double>(int,
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
                                                           double**);
template void linalg::cuda_partial_pivoting_solver<float>(int,
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
                                                          float**);
