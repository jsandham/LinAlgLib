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

#ifndef LINEAR_TYPES_H
#define LINEAR_TYPES_H

namespace linalg
{
    /**
     * @brief Opaque descriptor struct for CSR triangular solve analysis.
     *
     * This structure holds preprocessed analysis data for solving triangular systems
     * using sparse CSR matrices. It should be created via create_csrtrsv_descr(),
     * populated via csrtrsv_analysis(), and destroyed via destroy_csrtrsv_descr().
     * Users should not access its members directly.
     */
    struct csrtrsv_descr;

    /**
     * @brief Descriptor for CSR matrix-vector product operations.
     *
     * Holds preprocessing data for efficient CSR matrix-vector multiplication.
     * Created with create_csrmv_descr() and destroyed with destroy_csrmv_descr().
     */
    struct csrmv_descr;

    /**
     * @brief Descriptor for CSR matrix-matrix addition operations.
     *
     * Holds preprocessing data for computing \f$C = \alpha A + \beta B\f$ in CSR format.
     * This is a two-stage descriptor: first csrgeam_nnz() determines the sparsity pattern,
     * then csrgeam_solve() computes the values.
     * Created with create_csrgeam_descr() and destroyed with destroy_csrgeam_descr().
     */
    struct csrgeam_descr;

    /**
     * @brief Descriptor for CSR matrix-matrix multiplication operations.
     *
     * Holds preprocessing data for computing \f$C = \alpha A \cdot B + \beta D\f$ in CSR format.
     * This is a two-stage descriptor: first csrgemm_nnz() determines the sparsity pattern,
     * then csrgemm_solve() computes the values.
     * Created with create_csrgemm_descr() and destroyed with destroy_csrgemm_descr().
     */
    struct csrgemm_descr;

    /**
     * @brief Descriptor for CSR incomplete Cholesky (IC(0)) factorization.
     *
     * Holds preprocessing data for computing the incomplete Cholesky factorization
     * of a sparse matrix in CSR format. Created with create_csric0_descr() and
     * destroyed with destroy_csric0_descr().
     */
    struct csric0_descr;

    /**
     * @brief Descriptor for CSR incomplete LU (ILU(0)) factorization.
     *
     * Holds preprocessing data for computing the incomplete LU factorization
     * of a sparse matrix in CSR format. Created with create_csrilu0_descr() and
     * destroyed with destroy_csrilu0_descr().
     */
    struct csrilu0_descr;

    struct tridiagonal_descr;
} // namespace linalg
#endif // LINEAR_TYPES_H
