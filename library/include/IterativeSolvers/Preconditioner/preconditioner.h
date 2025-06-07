//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <vector>

#include "../../linalglib_export.h"
#include "../../csr_matrix.h"

#include "../AMG/amg.h"

/*! \file
 *  \brief preconditioner.h provides different preconditioners that can be used with Krylov solvers
 */

namespace linalg
{
/*! \ingroup iterative_solvers
 * \brief Abstract base class for preconditioners.
 *
 * \details
 * This abstract class defines the interface for preconditioners used in
 * iterative linear solvers. Preconditioners are used to improve the
 * convergence rate of iterative methods by transforming the original linear
 * system into one that is easier to solve. Derived classes implement specific
 * preconditioning techniques.
 */
class preconditioner
{
public:
    /*! \brief Default constructor. */
    preconditioner(){};
    /*! \brief Virtual destructor. */
    virtual ~preconditioner(){};

    /*! \brief Builds the preconditioner based on the sparse matrix.
    *
    * \param A The input matrix for which the preconditioner is to be built.
    */
    virtual void build(const csr_matrix& A) = 0;

    /*! \brief Solves the preconditioning system M*x = rhs.
    *
    * \param rhs Right-hand side vector of the preconditioning system.
    * \param x Output vector containing the solution of the preconditioning system.
    */
    virtual void solve(const vector<double>& rhs, vector<double>& x) const = 0;
};

/*! \ingroup iterative_solvers
 * \brief Jacobi preconditioner.
 *
 * \details
 * Implements the Jacobi preconditioning method. The Jacobi preconditioner
 * is a diagonal preconditioner where the preconditioning matrix M is the
 * diagonal of the original matrix A. The solve operation involves a simple
 * element-wise division by the diagonal entries.
 */
class jacobi_precond : public preconditioner
{
private:
    /*! \brief Stores the inverse of the diagonal elements of the matrix. */
    vector<double> diag;

public:
    /*! \brief Constructs a new `jacobi_precond` object. */
    jacobi_precond();
    /*! \brief Destroys the `jacobi_precond` object. */
    ~jacobi_precond();

    /*! \brief Builds the Jacobi preconditioner.
     *
     * This method computes and stores the inverse of the diagonal elements
     * of the input matrix `A`.
     * \param A The input matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$.
     *
     * This method applies the Jacobi preconditioning by performing an
     * element-wise division of the right-hand side vector `rhs` by the
     * stored inverse diagonal elements.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};

/*! \ingroup iterative_solvers
 * \brief Gauss-Seidel preconditioner.
 *
 * \details
 * Implements the Gauss-Seidel preconditioning method. The Gauss-Seidel
 * preconditioner uses the lower triangular part of the matrix A (including
 * the diagonal) as the preconditioning matrix M. The solve operation involves
 * a forward substitution.
 */
class gauss_seidel_precond : public preconditioner
{
private:
    /*! \brief Stores a copy of the matrix A.
     * \details The Gauss-Seidel preconditioner requires access to the
     * lower triangular part of the original matrix during the solve phase.
     */
    csr_matrix A;

public:
    /*! \brief Constructs a new `gauss_seidel_precond` object. */
    gauss_seidel_precond();

    /*! \brief Destroys the `gauss_seidel_precond` object. */
    ~gauss_seidel_precond();

    /*! \brief Builds the Gauss-Seidel preconditioner.
     *
     * This method essentially stores a copy of the input matrix `A`, as
     * the Gauss-Seidel solve operation requires the lower triangular
     * part of the original matrix.
     * \param A The input matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$ using forward substitution.
     *
     * This method applies the Gauss-Seidel preconditioning by performing
     * a forward substitution using the lower triangular part of the matrix `A`.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};

/*! \ingroup iterative_solvers
 * \brief Successive Over-Relaxation (SOR) preconditioner.
 *
 * \details
 * Implements the Successive Over-Relaxation (SOR) preconditioning method.
 * SOR is a variant of Gauss-Seidel that introduces a relaxation parameter
 * (\f$\omega\f$) to potentially accelerate convergence. The preconditioning matrix
 * M is related to the lower triangular part of A scaled by omega.
 */
class SOR_precond : public preconditioner
{
private:
    /*! \brief Stores a copy of the original matrix A. */
    csr_matrix A;
    /*! \brief The relaxation parameter (\f$\omega\f$) for SOR. */
    double omega;
    /*! \brief Stores the diagonal elements of the matrix A, often used in the SOR solve process. */
    vector<double> diag;

public:
    /*! \brief Constructs a new `SOR_precond` object with a specified relaxation parameter.
     * \param omega The relaxation parameter for the SOR method.
     */
    SOR_precond(double omega);

    /*! \brief Destroys the `SOR_precond` object. */
    ~SOR_precond();

    /*! \brief Builds the SOR preconditioner.
     *
     * This method typically involves storing the input matrix `A` and
     * potentially pre-calculating values like the inverse diagonal elements
     * that are used in the SOR solve.
     * \param A The input matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$ using the SOR method.
     *
     * This method applies the SOR preconditioning iteration to solve for `x`.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};

/*! \ingroup iterative_solvers
 * \brief Symmetric Gauss-Seidel preconditioner.
 *
 * \details
 * Implements the Symmetric Gauss-Seidel (SGS) preconditioning method.
 * SGS applies a forward Gauss-Seidel sweep followed by a backward
 * Gauss-Seidel sweep as the preconditioning operation.
 */
class symmetric_gauss_seidel_precond : public preconditioner
{
private:
    /*! \brief Stores a copy of the original matrix A. */
    csr_matrix A;
    /*! \brief Stores the diagonal elements of the matrix A, used in the SGS solve process. */
    vector<double> diag;

public:
    /*! \brief Constructs a new `symmetric_gauss_seidel_precond` object. */
    symmetric_gauss_seidel_precond();

    /*! \brief Destroys the `symmetric_gauss_seidel_precond` object. */
    ~symmetric_gauss_seidel_precond();

    /*! \brief Builds the Symmetric Gauss-Seidel preconditioner.
     *
     * This method typically involves storing the input matrix `A` and
     * pre-calculating the diagonal elements needed for the SGS sweeps.
     * \param A The input matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$ using Symmetric Gauss-Seidel.
     *
     * This method applies a forward Gauss-Seidel sweep followed by a
     * backward Gauss-Seidel sweep to compute the preconditioned result `x`.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};

/*! \ingroup iterative_solvers
 * \brief Incomplete LU (ILU) preconditioner.
 *
 * \details
 * Implements the Incomplete LU (ILU) factorization preconditioner. ILU
 * computes an approximate LU factorization of the matrix A, where L is a
 * lower triangular matrix and U is an upper triangular matrix, with a certain
 * fill-in pattern. The solve operation involves a forward solve with L and
 * a backward solve with U. The specific ILU variant (e.g., ILU(0), ILUT)
 * is determined by the implementation of the build method.
 */
class ilu_precond : public preconditioner
{
private:
    /*! \brief Stores the approximate LU factorization of the matrix A.
     * \details This `csr_matrix` will typically store the non-zero elements
     * of both L (lower triangular) and U (upper triangular) factors,
     * potentially in a merged format.
     */
    csr_matrix LU;

public:
    /*! \brief Constructs a new `ilu_precond` object. */
    ilu_precond();

    /*! \brief Destroys the `ilu_precond` object. */
    ~ilu_precond();

    /*! \brief Builds the Incomplete LU (ILU) preconditioner.
     *
     * This method computes the approximate LU factorization of the input matrix `A`.
     * The specific ILU variant (e.g., ILU(0), ILUT with a drop tolerance)
     * is determined by the internal implementation of this method.
     * \param A The input matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$ using the ILU factors.
     *
     * This method performs a forward solve with the L factor, followed by a
     * backward solve with the U factor to compute the preconditioned result `x`.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};

/*! \ingroup iterative_solvers
 * \brief Incomplete Cholesky (IC) preconditioner.
 *
 * \details
 * Implements the Incomplete Cholesky (IC) factorization preconditioner. IC
 * computes an approximate Cholesky factorization of a symmetric positive
 * definite matrix A, where L is a lower triangular matrix such that
 * \f$A \approx LL^T\f$. The solve operation involves a forward solve with L
 * and a backward solve with \f$L^T\f$. The specific IC variant (e.g., IC(0), ICT)
 * is determined by the implementation of the build method.
 */
class ic_precond : public preconditioner
{
private:
    /*! \brief Stores the approximate Cholesky factor L (and implicitly L^T) of the matrix A.
     * \details For a symmetric positive definite matrix, the `LLT` matrix
     * will store the lower triangular factor L.
     */
    csr_matrix LLT;

public:
    /*! \brief Constructs a new `ic_precond` object. */
    ic_precond();

    /*! \brief Destroys the `ic_precond` object. */
    ~ic_precond();

    /*! \brief Builds the Incomplete Cholesky (IC) preconditioner.
     *
     * This method computes the approximate Cholesky factorization of the
     * input symmetric positive definite matrix `A`. The specific IC variant
     * (e.g., IC(0), ICT with a drop tolerance) is determined by the
     * internal implementation of this method.
     * \param A The input symmetric positive definite matrix for which the preconditioner is to be built.
     */
    void build(const csr_matrix& A) override;

    /*! \brief Solves the preconditioning system \f$M \cdot x = \text{rhs}\f$ using the IC factors.
     *
     * This method performs a forward solve with the L factor, followed by a
     * backward solve with the \f$L^T\f$ factor to compute the preconditioned result `x`.
     *
     * \param rhs The right-hand side vector.
     * \param x The output vector, which will contain the preconditioned result.
     */
    void solve(const vector<double>& rhs, vector<double>& x) const override;
};
}

#endif