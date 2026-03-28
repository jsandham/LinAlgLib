#include <cmath>

#include "../../trace.h"
#include "host_axpy.h"

namespace linalg
{
    template <typename T>
    static void host_axpy_impl(int n, T alpha, const T* x, T* y)
    {
        ROUTINE_TRACE("host_axpy_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            y[i] = alpha == 1.0 ? x[i] + y[i] : alpha * x[i] + y[i];
        }
    }

    template <typename T>
    static void host_axpby_impl(int n, T alpha, const T* x, T beta, T* y)
    {
        ROUTINE_TRACE("host_axpby_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            if(alpha == 0.0)
            {
                y[i] = beta * y[i];
            }
            else if(beta == 0.0)
            {
                y[i] = alpha * x[i];
            }
            else
            {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }
    }

    template <typename T>
    static void host_axpbypgz_impl(int n, T alpha, const T* x, T beta, const T* y, T gamma, T* z)
    {
        ROUTINE_TRACE("host_axpbypgz_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            z[i] = alpha * x[i] + beta * y[i] + gamma * z[i];
        }
    }

    template <typename T>
    static T host_dot_product_impl(const T* x, const T* y, int n)
    {
        ROUTINE_TRACE("host_dot_product_impl");
        T dot_prod = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : dot_prod)
#endif
        for(int i = 0; i < n; i++)
        {
            dot_prod += x[i] * y[i];
        }
        return dot_prod;
    }
}

void linalg::host_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_axpy");
    host_axpy_impl(x.get_size(), alpha, x.get_vec(), y.get_vec());
}

void linalg::host_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_axpby");
    host_axpby_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec());
}

void linalg::host_axpbypgz(double                alpha,
                           const vector<double>& x,
                           double                beta,
                           const vector<double>& y,
                           double                gamma,
                           vector<double>&       z)
{
    ROUTINE_TRACE("linalg::host_axpbypgz");
    host_axpbypgz_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec());
}

double linalg::host_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::host_dot_product");
    return host_dot_product_impl(x.get_vec(), y.get_vec(), x.get_size());
}
