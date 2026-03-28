#include <cmath>

#include "../../trace.h"
#include "host_matrix_vector.h"

namespace linalg
{
    template <typename T>
    static void host_compute_residual_impl(const int* csr_row_ptr,
                                           const int* csr_col_ind,
                                           const T*   csr_val,
                                           const T*   x,
                                           const T*   b,
                                           T*         res,
                                           int        n)
    {
        ROUTINE_TRACE("host_compute_residual_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            T s = 0.0;
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                s += csr_val[j] * x[csr_col_ind[j]];
            }
            res[i] = b[i] - s;
        }
    }

    template <typename T>
    static void host_csrmv_impl(int        m,
                                int        n,
                                int        nnz,
                                T          alpha,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                const T*   csr_val,
                                const T*   x,
                                T          beta,
                                T*         y)
    {
        ROUTINE_TRACE("host_csrmv");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m; i++)
        {
            double s = 0.0;
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                s = std::fma(csr_val[j], x[csr_col_ind[j]], s);
            }
            y[i] = beta == 0.0 ? alpha * s : std::fma(alpha, s, beta * y[i]);
        }
    }
}

void linalg::host_compute_residual(const csr_matrix&     A,
                                   const vector<double>& x,
                                   const vector<double>& b,
                                   vector<double>&       res)
{
    ROUTINE_TRACE("linalg::host_compute_residual");
    host_compute_residual_impl(A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               x.get_vec(),
                               b.get_vec(),
                               res.get_vec(),
                               A.get_m());
}

void linalg::host_csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr) {}

void linalg::host_csrmv_solve(double                alpha,
                              const csr_matrix&     A,
                              const vector<double>& x,
                              double                beta,
                              vector<double>&       y,
                              csrmv_algorithm       alg,
                              const csrmv_descr*    descr)
{
    ROUTINE_TRACE("linalg::host_csrmv_solve");
    switch(alg)
    {
    case csrmv_algorithm::default_algorithm:
    case csrmv_algorithm::merge_path:
    case csrmv_algorithm::rowsplit:
    case csrmv_algorithm::nnzsplit:
        host_csrmv_impl(A.get_m(),
                        A.get_n(),
                        A.get_nnz(),
                        alpha,
                        A.get_row_ptr(),
                        A.get_col_ind(),
                        A.get_val(),
                        x.get_vec(),
                        beta,
                        y.get_vec());
        break;
    default:
        break;
    }
}
