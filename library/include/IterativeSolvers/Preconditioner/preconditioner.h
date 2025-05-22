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

#include "../AMG/amg.h"

/*! \file
 *  \brief preconditioner.h provides different preconditioners that can be used with Krylov solvers
 */

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
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    virtual void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) = 0;

    /*! \brief Solves the preconditioning system M*x = rhs.
    *
    * \param rhs Right-hand side vector of the preconditioning system.
    * \param x Output vector containing the solution of the preconditioning system.
    * \param n Size of the vectors.
    */
    virtual void solve(const double* rhs, double* x, int n) const = 0;
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
    /*! \brief Vector storing the inverse of the diagonal elements. */
    std::vector<double> m_diag;

public:
    /*! \brief Default constructor. */
    jacobi_precond();
    /*! \brief Default destructor. */
    ~jacobi_precond();

    /*! \brief Builds the Jacobi preconditioner by storing the inverse of the diagonal elements.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the Jacobi preconditioning system M*x = rhs by element-wise division.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
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
    /*! \brief Pointer to the CSR row pointer array of the matrix. */
    const int* m_csr_row_ptr;
    /*! \brief Pointer to the CSR column index array of the matrix. */
    const int* m_csr_col_ind;
    /*! \brief Pointer to the CSR value array of the matrix. */
    const double* m_csr_val;

public:
    /*! \brief Default constructor. */
    gauss_seidel_precond();
    /*! \brief Default destructor. */
    ~gauss_seidel_precond();

    /*! \brief Builds the Gauss-Seidel preconditioner by storing references to the matrix data.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the Gauss-Seidel preconditioning system M*x = rhs using forward substitution.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
};

/*! \ingroup iterative_solvers
* \brief Successive Over-Relaxation (SOR) preconditioner.
*
* \details
* Implements the Successive Over-Relaxation (SOR) preconditioning method.
* SOR is a variant of Gauss-Seidel that introduces a relaxation parameter
* (omega) to potentially accelerate convergence. The preconditioning matrix
* M is related to the lower triangular part of A scaled by omega.
*/
class SOR_precond : public preconditioner
{
private:
    /*! \brief Pointer to the CSR row pointer array of the matrix. */
    const int* m_csr_row_ptr;
    /*! \brief Pointer to the CSR column index array of the matrix. */
    const int* m_csr_col_ind;
    /*! \brief Pointer to the CSR value array of the matrix. */
    const double* m_csr_val;
    /*! \brief Relaxation parameter omega. */
    double m_omega;
    /*! \brief Vector storing the diagonal elements. */
    std::vector<double> m_diag;

public:
    /*! \brief Constructor with the relaxation parameter.
    * \param omega The relaxation parameter for SOR.
    */
    SOR_precond(double omega);
    /*! \brief Default destructor. */
    ~SOR_precond();

    /*! \brief Builds the SOR preconditioner by storing references to the matrix data and the diagonal.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the SOR preconditioning system M*x = rhs using forward substitution with relaxation.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
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
    /*! \brief Pointer to the CSR row pointer array of the matrix. */
    const int* m_csr_row_ptr;
    /*! \brief Pointer to the CSR column index array of the matrix. */
    const int* m_csr_col_ind;
    /*! \brief Pointer to the CSR value array of the matrix. */
    const double* m_csr_val;
    /*! \brief Vector storing the diagonal elements. */
    std::vector<double> m_diag;

public:
    /*! \brief Default constructor. */
    symmetric_gauss_seidel_precond();
    /*! \brief Default destructor. */
    ~symmetric_gauss_seidel_precond();

    /*! \brief Builds the Symmetric Gauss-Seidel preconditioner by storing references to the matrix data and the diagonal.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the Symmetric Gauss-Seidel preconditioning system M*x = rhs using a forward and backward Gauss-Seidel sweep.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
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
    /*! \brief CSR row pointer for the LU factors. */
    std::vector<int> m_csr_row_ptr_LU;
    /*! \brief CSR column indices for the LU factors. */
    std::vector<int> m_csr_col_ind_LU;
    /*! \brief CSR values for the LU factors. */
    std::vector<double> m_csr_val_LU;

public:
    /*! \brief Default constructor. */
    ilu_precond();
    /*! \brief Default destructor. */
    ~ilu_precond();

    /*! \brief Builds the ILU preconditioner by computing the incomplete LU factorization.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the ILU preconditioning system M*x = rhs using forward and backward substitution with the LU factors.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
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
    /*! \brief CSR row pointer for the LLT factors. */
    std::vector<int> m_csr_row_ptr_LLT;
    /*! \brief CSR column indices for the LLT factors. */
    std::vector<int> m_csr_col_ind_LLT;
    /*! \brief CSR values for the LLT factors. */
    std::vector<double> m_csr_val_LLT;

public:
    /*! \brief Default constructor. */
    ic_precond();
    /*! \brief Default destructor. */
    ~ic_precond();

    /*! \brief Builds the IC preconditioner by computing the incomplete Cholesky factorization.
    *
    * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
    * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
    * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
    * \param m Number of rows in the matrix.
    * \param n Number of columns in the matrix.
    * \param nnz Number of non-zero elements in the matrix.
    */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the IC preconditioning system M*x = rhs using forward and backward substitution with the LLT factors.
    *
    * \param rhs Right-hand side vector.
    * \param x Output vector containing the preconditioned result.
    * \param n Size of the vectors.
    */
    void solve(const double* rhs, double* x, int n) const override;
};

/*! \ingroup iterative_solvers
* \brief Smoothed Aggregation Algebraic Multigrid (SA-AMG) preconditioner.
*
* \details
* Implements a preconditioner based on the Smoothed Aggregation Algebraic
* Multigrid (SA-AMG) method. This preconditioner internally uses a
* multigrid hierarchy constructed via SA-AMG to perform the preconditioning
* solve. The solve operation involves applying one or more multigrid cycles.
* This approach can provide efficient preconditioning for a wide range of
* sparse linear systems, especially those arising from the discretization
* of partial differential equations.
*
* The \p build method of this class sets up the AMG hierarchy using the
* provided matrix. This involves creating a series of coarser grids and
* the corresponding transfer operators (prolongation and restriction).
* The \p solve method then performs one multigrid cycle (V-cycle, W-cycle, etc.)
* using the constructed hierarchy to approximate the inverse of the system
* matrix applied to the right-hand side vector.
*
* The parameters \p presmoothing, \p postsmoothing, \p cycle, and \p smoother
* control the behavior of the multigrid cycle used within the preconditioner.
* These should be chosen based on the properties of the linear system being solved.
*/
class saamg_precond : public preconditioner
{
private:
    /*! \brief Hierarchy of restriction, prolongation, and coarse grid operators. */
    heirarchy m_hierachy;
    /*! \brief Number of pre-smoothing steps in the AMG cycle. */
    int m_presmoothing;
    /*! \brief Number of post-smoothing steps in the AMG cycle. */
    int m_postsmoothing;
    /*! \brief Type of AMG cycle to use (\ref Cycle). */
    Cycle m_cycle;
    /*! \brief Type of smoother to use in the AMG cycle (\ref Smoother). */
    Smoother m_smoother;

public:
    /*! \brief Constructor with AMG parameters.
     * \param presmoothing Number of pre-smoothing steps to perform in each AMG cycle during the solve phase.
     * \param postsmoothing Number of post-smoothing steps to perform in each AMG cycle during the solve phase.
     * \param cycle The type of AMG cycle to use during the solve phase (\ref Cycle).
     * \param smoother The type of smoother to use at each level of the AMG hierarchy during the solve phase (\ref Smoother).
     */
    saamg_precond(int presmoothing, int postsmoothing, Cycle cycle, Smoother smoother);
    /*! \brief Default destructor. */
    ~saamg_precond();

    /*! \brief Builds the SA-AMG preconditioner by setting up the multigrid hierarchy.
     *
     * This method calls an SA-AMG setup routine (e.g., \ref saamg_setup) to
     * construct the hierarchy of coarse grids and transfer operators based on
     * the input sparse matrix.
     *
     * \param csr_row_ptr Array of \p m+1 elements pointing to the start of each row in CSR format.
     * \param csr_col_ind Array of \p nnz elements containing column indices in CSR format.
     * \param csr_val Array of \p nnz elements containing non-zero values in CSR format.
     * \param m Number of rows in the matrix.
     * \param n Number of columns in the matrix.
     * \param nnz Number of non-zero elements in the matrix.
     */
    void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;

    /*! \brief Solves the preconditioning system M*x = rhs by performing one AMG cycle.
     *
     * This method applies one AMG cycle (of the type specified in the constructor)
     * using the pre-computed hierarchy to approximate the solution of the
     * preconditioning system.
     *
     * \param rhs Right-hand side vector.
     * \param x Output vector containing the preconditioned result.
     * \param n Size of the vectors.
     */
    void solve(const double* rhs, double* x, int n) const override;
};

#endif