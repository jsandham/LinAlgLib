#include "device_tridiagonal.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_tridiagonal.h"
#endif

void linalg::allocate_tridiagonal_device_data(tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_tridiagonal_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_tridiagonal_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::free_tridiagonal_device_data(tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::free_tridiagonal_device_data");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_tridiagonal_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_tridiagonal_analysis(int                  m,
                                         int                  n,
                                         const vector<float>& lower_diag,
                                         const vector<float>& main_diag,
                                         const vector<float>& upper_diag,
                                         tridiagonal_descr*   descr)
{
    ROUTINE_TRACE("linalg::device_tridiagonal_analysis");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_tridiagonal_analysis(
            m, n, lower_diag.get_vec(), main_diag.get_vec(), upper_diag.get_vec(), descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_tridiagonal_solver(int                      m,
                                       int                      n,
                                       const vector<float>&     lower_diag,
                                       const vector<float>&     main_diag,
                                       const vector<float>&     upper_diag,
                                       const vector<float>&     b,
                                       vector<float>&           x,
                                       const tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::device_tridiagonal_solver");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_tridiagonal_solver(m,
                                          n,
                                          lower_diag.get_vec(),
                                          main_diag.get_vec(),
                                          upper_diag.get_vec(),
                                          b.get_vec(),
                                          x.get_vec(),
                                          descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
