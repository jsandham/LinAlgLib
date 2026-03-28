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
