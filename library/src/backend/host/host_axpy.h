#ifndef HOST_AXPY_H
#define HOST_AXPY_H

#include "vector.h"

namespace linalg
{
    void   host_axpy(double alpha, const vector<double>& x, vector<double>& y);
    void   host_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y);
    void   host_axpbypgz(double                alpha,
                         const vector<double>& x,
                         double                beta,
                         const vector<double>& y,
                         double                gamma,
                         vector<double>&       z);
    double host_dot_product(const vector<double>& x, const vector<double>& y);
}

#endif
