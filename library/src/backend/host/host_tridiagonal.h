#ifndef HOST_TRIDIAGONAL_H
#define HOST_TRIDIAGONAL_H

#include "vector.h"

namespace linalg
{
    struct tridiagonal_descr;

    void host_tridiagonal_analysis(int                  m,
                                   int                  n,
                                   const vector<float>& lower_diag,
                                   const vector<float>& main_diag,
                                   const vector<float>& upper_diag,
                                   tridiagonal_descr*   descr);
    void host_tridiagonal_solver(int                      m,
                                 int                      n,
                                 const vector<float>&     lower_diag,
                                 const vector<float>&     main_diag,
                                 const vector<float>&     upper_diag,
                                 const vector<float>&     b,
                                 vector<float>&           x,
                                 const tridiagonal_descr* descr);
}

#endif
