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
    void cuda_partial_pivoting_solver(int           m,
                                      int           n,
                                      const double* lower_diag,
                                      const double* main_diag,
                                      const double* upper_diag,
                                      const double* B,
                                      double*       X,
                                      double**      lower_pad,
                                      double**      main_pad,
                                      double**      upper_pad,
                                      double**      B_pad,
                                      double**      w_pad,
                                      double**      v_pad,
                                      double**      mt,
                                      double**      S_lower,
                                      double**      S_main,
                                      double**      S_upper,
                                      double**      S_B);

    void cuda_non_pivoting_solver(int           m,
                                  int           n,
                                  const double* lower_diag,
                                  const double* main_diag,
                                  const double* upper_diag,
                                  const double* B,
                                  double*       X,
                                  double**      lower_modified,
                                  double**      main_modified,
                                  double**      upper_modified,
                                  double**      B_modified,
                                  double**      spike_lower,
                                  double**      spike_main,
                                  double**      spike_upper,
                                  double**      spike_B,
                                  double**      spike_X);
}

#endif
