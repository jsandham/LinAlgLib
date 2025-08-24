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
#include "device_amg_strength.h"
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_amg_strength.h"
#endif

void linalg::device_compute_strong_connections(const csr_matrix& A,
                                               double            eps,
                                               vector<int>&      connections)
{
    ROUTINE_TRACE("linalg::device_compute_strong_connections");

    if constexpr(is_cuda_available())
    {
        // Extract diagaonl
        //vector<double> diag(A.get_m());
        //A.extract_diagonal(diag);

        CALL_CUDA(cuda_compute_strong_connections(A.get_m(),
                                                  A.get_n(),
                                                  A.get_nnz(),
                                                  A.get_row_ptr(),
                                                  A.get_col_ind(),
                                                  A.get_val(),
                                                  eps,
                                                  connections.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // // Extract diagaonl
    // //vector<double> diag(A.get_m());
    // //A.extract_diagonal(diag);

    // device_compute_strong_connections_impl(A.get_m(),
    //                                        A.get_n(),
    //                                        A.get_nnz(),
    //                                        A.get_row_ptr(),
    //                                        A.get_col_ind(),
    //                                        A.get_val(),
    //                                        eps,
    //                                        connections.get_vec());
}

void linalg::device_compute_classical_strong_connections(const csr_matrix& A,
                                                         double            theta,
                                                         csr_matrix&       S,
                                                         vector<int>&      connections)
{
    ROUTINE_TRACE("linalg::device_compute_classical_strong_connections");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_compute_classical_strong_connections(A.get_m(),
                                                            A.get_n(),
                                                            A.get_nnz(),
                                                            A.get_row_ptr(),
                                                            A.get_col_ind(),
                                                            A.get_val(),
                                                            theta,
                                                            S.get_row_ptr(),
                                                            S.get_col_ind(),
                                                            S.get_val(),
                                                            connections.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // device_compute_classical_strong_connections_impl(A.get_m(),
    //                                                  A.get_n(),
    //                                                  A.get_nnz(),
    //                                                  A.get_row_ptr(),
    //                                                  A.get_col_ind(),
    //                                                  A.get_val(),
    //                                                  theta,
    //                                                  S.get_row_ptr(),
    //                                                  S.get_col_ind(),
    //                                                  S.get_val(),
    //                                                  connections.get_vec());
}