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
#ifndef CUDA_TRIDIAGONAL_H
#define CUDA_TRIDIAGONAL_H

namespace linalg
{
    template <typename T>
    void cuda_partial_pivoting_solver(int      m,
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
                                      T**      S_B);

    template <typename T>
    void cuda_non_pivoting_solver(int      m,
                                  int      n,
                                  const T* lower_diag,
                                  const T* main_diag,
                                  const T* upper_diag,
                                  const T* B,
                                  T*       X,
                                  T**      lower_modified,
                                  T**      main_modified,
                                  T**      upper_modified,
                                  T**      B_modified,
                                  T**      spike_lower,
                                  T**      spike_main,
                                  T**      spike_upper,
                                  T**      spike_B,
                                  T**      spike_X);
}

#endif
