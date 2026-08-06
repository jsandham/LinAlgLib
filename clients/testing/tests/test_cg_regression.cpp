#include <gtest/gtest.h>

#include "linalg.h"

TEST(CGRegression, RespectsMaxIterations)
{
    std::vector<int>    row_ptr = {0, 1, 2};
    std::vector<int>    col_ind = {0, 1};
    std::vector<double> values  = {4.0, 4.0};

    linalg::csr_matrix A(row_ptr, col_ind, values, 2, 2, 2);

    linalg::vector<double> x(2);
    x.zeros();

    std::vector<double>    b_data = {1.0, 2.0};
    linalg::vector<double> b(b_data);

    linalg::iter_control control;
    control.max_iter = 1;
    control.abs_tol  = 1e-14;
    control.rel_tol  = 1e-14;

    linalg::cg_solver solver;
    solver.build(A);

    int iterations = solver.solve(A, x, b, nullptr, control);

    EXPECT_EQ(iterations, 1);
    EXPECT_NEAR(x[0], 0.25, 1e-12);
    EXPECT_NEAR(x[1], 0.5, 1e-12);
}
