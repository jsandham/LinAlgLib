//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#ifndef __POWER_ITERATION_H__
#define __POWER_ITERATION_H__

#include "../linalg_export.h"

/*! \ingroup iterative_solvers
 * \brief Power iteration to find the dominant eigenvalue and corresponding eigenvector.
 *
 * \details
 * \p power_iteration implements the power iteration algorithm to find the
 * eigenvalue with the largest magnitude (the dominant eigenvalue) and its
 * corresponding eigenvector for a given sparse matrix. The algorithm
 * iteratively multiplies a starting vector by the matrix and normalizes the
 * result. This process converges to the eigenvector associated with the
 * dominant eigenvalue. The eigenvalue is then estimated using the Rayleigh
 * quotient.
 *
 * **Algorithm Overview:**
 *
 * 1. Initialize: Choose a random starting vector \p eigenVec of size \p n.
 * Normalize it (e.g., to have a Euclidean norm of 1).
 * 2. Iterate up to \p max_iter times:
 * a. Multiply the current eigenvector by the sparse matrix: \f$y = A \cdot \text{eigenVec}\f$.
 * b. Calculate the Rayleigh quotient to estimate the eigenvalue:
 * \f$\lambda_{new} = (\text{eigenVec}^T y) / (\text{eigenVec}^T \text{eigenVec})\f$.
 * c. Normalize the resulting vector: \f$\text{eigenVec}_{new} = y / ||y||_2\f$.
 * d. Check for convergence: If the absolute difference between the current
 * eigenvalue estimate and the previous one is less than \p tol, or if
 * the maximum number of iterations \p max_iter is reached, then stop.
 * e. Update the eigenvector: \f$\text{eigenVec} = \text{eigenVec}_{new}\f$.
 * f. Store the current eigenvalue estimate.
 * 3. Return the last calculated eigenvalue.
 *
 * **Note:** The power iteration method converges to the dominant eigenvalue
 * and its eigenvector. If there are multiple eigenvalues with the same largest
 * magnitude, the convergence might be to a linear combination of their
 * eigenvectors. The rate of convergence depends on the ratio of the magnitude
 * of the dominant eigenvalue to the magnitude of the second largest eigenvalue.
 * A larger ratio leads to faster convergence.
 *
 * @param[in] csr_row_ptr
 * Array of \p n+1 elements that point to the start of every row of
 * the input sparse matrix in CSR format.
 * @param[in] csr_col_ind
 * Array of \p nnz elements containing the column indices of the
 * non-zero entries in the input sparse matrix (CSR format).
 * @param[in] csr_val
 * Array of \p nnz elements containing the numerical values of the
 * non-zero entries in the input sparse matrix (CSR format).
 * @param[inout] eigenVec
 * Array of \p n elements serving as the initial guess for the eigenvector.
 * On output, if the algorithm converges, this array will contain the
 * eigenvector corresponding to the dominant eigenvalue. It should be
 * initialized with a non-zero vector.
 * @param[in] tol
 * The stopping tolerance for the eigenvalue convergence. The algorithm
 * terminates when the absolute difference between successive eigenvalue
 * estimates is less than this value.
 * @param[in] n
 * The dimension of the square sparse CSR matrix (number of rows or columns)
 * and the size of the eigenvector.
 * @param[in] max_iter
 * The maximum number of iterations allowed for the power iteration algorithm
 * to converge. The algorithm will terminate if convergence is not reached
 * within this number of iterations.
 *
 * @retval double
 * The dominant eigenvalue (the eigenvalue with the largest magnitude) of the
 * input sparse matrix.
 *
 * \par Example
 * \code{.cpp}
 * #include <vector>
 * #include <iostream>
 * #include <cmath>
 * #include "linalg.h"
 *
 * int main() {
 * int m, n_local, nnz_local;
 * std::vector<int> csr_row_ptr_local;
 * std::vector<int> csr_col_ind_local;
 * std::vector<double> csr_val_local;
 * const char* matrix_file = "dummy_matrix.mtx";
 * load_mtx_file(matrix_file, csr_row_ptr_local, csr_col_ind_local, csr_val_local, m, n_local, nnz_local);
 *
 * std::vector<double> eigenvector(n_local, 1.0);
 * double tolerance = 1e-6;
 * int max_iterations = 100;
 *
 * double eigenvalue = power_iteration(csr_row_ptr_local.data(), csr_col_ind_local.data(), csr_val_local.data(),
 * eigenvector.data(), tolerance, n_local, max_iterations);
 *
 * std::cout << "Dominant Eigenvalue: " << eigenvalue << std::endl;
 * std::cout << "Corresponding Eigenvector: ";
 * for (double val : eigenvector) {
 * std::cout << val << " ";
 * }
 * std::cout << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
double power_iteration(const int*    csr_row_ptr,
                       const int*    csr_col_ind,
                       const double* csr_val,
                       double*       eigenVec,
                       const double  tol,
                       const int     n,
                       const int     max_iter);

#endif
