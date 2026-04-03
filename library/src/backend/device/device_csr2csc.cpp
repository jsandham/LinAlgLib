//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#include "device_csr2csc.h"
#include "device_memory.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_csr2csc.h"
#include "cuda/cuda_memory.h"
#endif

void linalg::device_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::device_transpose_matrix");
    if constexpr(is_cuda_available())
    {
        transposeA.resize(A.get_n(), A.get_m(), A.get_nnz());

        size_t buffer_size;
        CALL_CUDA(cuda_csr2csc_buffer_size(A.get_m(),
                                           A.get_n(),
                                           A.get_nnz(),
                                           A.get_row_ptr(),
                                           A.get_col_ind(),
                                           A.get_val(),
                                           &buffer_size));

        unsigned char* buffer = nullptr;
        CALL_CUDA(cuda_allocate(&buffer, buffer_size));

        CALL_CUDA(cuda_csr2csc(A.get_m(),
                               A.get_n(),
                               A.get_nnz(),
                               A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               transposeA.get_row_ptr(),
                               transposeA.get_col_ind(),
                               transposeA.get_val(),
                               buffer));

        CALL_CUDA(cuda_free(buffer));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
