#include "device_axpy.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_axpy.h"
#endif

void linalg::device_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpy");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpy(x.get_size(), alpha, x.get_vec(), y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpby");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpby(x.get_size(), alpha, x.get_vec(), beta, y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_axpbypgz(double                alpha,
                             const vector<double>& x,
                             double                beta,
                             const vector<double>& y,
                             double                gamma,
                             vector<double>&       z)
{
    ROUTINE_TRACE("linalg::device_axpbypgz");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_axpbypgz(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

double linalg::device_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_dot_product");
    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_dot_product(x.get_vec(), y.get_vec(), x.get_size()));
    }
    std::cout << "Error: Not device backend available for the function " << __func__ << std::endl;
    return 0.0;
}
