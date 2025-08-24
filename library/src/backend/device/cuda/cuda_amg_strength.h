//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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
#ifndef CUDA_AMG_AGGREGATION_H
#define CUDA_AMG_AGGREGATION_H

namespace linalg
{
    void cuda_compute_strong_connections(int           m,
                                         int           n,
                                         int           nnz,
                                         const int*    csr_row_ptr,
                                         const int*    csr_col_ind,
                                         const double* csr_val,
                                         double        eps,
                                         int*          connections);

    void cuda_compute_classical_strong_connections(int           m,
                                                   int           n,
                                                   int           nnz,
                                                   const int*    csr_row_ptr_A,
                                                   const int*    csr_col_ind_A,
                                                   const double* csr_val_A,
                                                   double        theta,
                                                   int*          csr_row_ptr_S,
                                                   int*          csr_col_ind_S,
                                                   double*       csr_val_S,
                                                   int*          connections);
}

#endif