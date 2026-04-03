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
    struct tridiagonal_descr;

    void free_tridiagonal_cuda_data(tridiagonal_descr* descr);

    void cuda_tridiagonal_analysis(int                m,
                                   int                n,
                                   const float*       lower_diag,
                                   const float*       main_diag,
                                   const float*       upper_diag,
                                   tridiagonal_descr* descr);

    void cuda_tridiagonal_solver(int                      m,
                                 int                      n,
                                 const float*             lower_diag,
                                 const float*             main_diag,
                                 const float*             upper_diag,
                                 const float*             b,
                                 float*                   x,
                                 const tridiagonal_descr* descr);
}

#endif
