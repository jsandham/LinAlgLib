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

#include "cuda_tridiagonal_spike_solver.h"
#include "cuda_tridiagonal_spike_solver_small.h"

#include "tridiagonal_spike_kernels.cuh"
#include "tridiagonal_spike_one_thread_per_system_kernels.cuh"

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
    template <typename T>
    static void tridiagonal_partial_pivoting_solver_dispatch(int      m,
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
                                                             int      level = 0)
    {
        ROUTINE_TRACE("tridiagonal_partial_pivoting_solver_dispatch");

        if(m <= 1024)
        {
            tridiagonal_spike_solver_small(m,
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
                                           mt_pad,
                                           S_lower,
                                           S_main,
                                           S_upper,
                                           S_B,
                                           level);
        }
        else
        {
            tridiagonal_spike_solver(m,
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
                                     mt_pad,
                                     S_lower,
                                     S_main,
                                     S_upper,
                                     S_B,
                                     level);
        }
    }
}

template <typename T>
void linalg::cuda_partial_pivoting_solver(int      m,
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
                                          T**      mt,
                                          T**      S_lower,
                                          T**      S_main,
                                          T**      S_upper,
                                          T**      S_B)
{
    ROUTINE_TRACE("cuda_partial_pivoting_solver");

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
