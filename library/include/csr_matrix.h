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

#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <vector>
#include <string>

#include "vector.h"

/*! \file
 *  \brief csr_matrx.h provides class for CSR sparse matrices
 */

namespace linalg
{
/*! \ingroup iterative_solvers
 * \brief Data class for storing sparse CSR matrices
 *
 * \details
 * This class represents a sparse matrix stored in the **Compressed Sparse Row (CSR)** format.
 * The CSR format is efficient for storing and operating on sparse matrices, as it only
 * stores the non-zero elements along with information about their row and column indices.
 */
class csr_matrix
{
private:
    /**
     * @brief Number of rows in the CSR matrix.
     */
    int m;

    /**
     * @brief Number of columns in the CSR matrix.
     */
    int n;

    /**
     * @brief Number of non-zeros in the CSR matrix.
     */
    int nnz;

    /**
     * @brief Row pointer array of CSR format.
     *
     * `hcsr_row_ptr` is an array of length `m + 1`. `hcsr_row_ptr[i]` stores the index
     * in the `csr_col_ind` and `csr_val` arrays where the non-zero elements of the
     * `i`-th row start. The last element, `hcsr_row_ptr[m]`, stores the total number
     * of non-zero elements (`nnz`).
     */
    std::vector<int> hcsr_row_ptr;

    /**
     * @brief Column indices array of CSR format.
     *
     * `hcsr_col_ind` is an array of length `nnz` that stores the column index of each
     * non-zero element. The column indices for the non-zero elements in row `i` are
     * stored in `hcsr_col_ind[hcsr_row_ptr[i] : hcsr_row_ptr[i+1] - 1]`.
     */
    std::vector<int> hcsr_col_ind;

    /**
     * @brief Values array of CSR format.
     *
     * `hcsr_val` is an array of length `nnz` that stores the numerical value of each
     * non-zero element. The values for the non-zero elements in row `i` are stored
     * in `hcsr_val[hcsr_row_ptr[i] : hcsr_row_ptr[i+1] - 1]`.
     */
    std::vector<double> hcsr_val;

    /*! \brief Flag indicating if the matrix data is currently on the host (CPU) or device (GPU). */
    bool on_host;

public:
    /*! \brief Default constructor.
     * Initializes an empty CSR matrix with zero dimensions and no non-zero elements.
     */
    csr_matrix();

    /*! \brief Constructor to initialize a CSR matrix with provided data.
     *
     * \param csr_row_ptr A `std::vector` containing the row pointer array.
     * \param csr_col_ind A `std::vector` containing the column indices array.
     * \param csr_val A `std::vector` containing the non-zero values array.
     * \param m The number of rows in the matrix.
     * \param n The number of columns in the matrix.
     * \param nnz The total number of non-zero elements in the matrix.
     */
    csr_matrix(const std::vector<int>& csr_row_ptr, const std::vector<int>& csr_col_ind, const std::vector<double>& csr_val, int m, int n, int nnz);

    /*! \brief Destructor.
     * Cleans up any resources allocated by the CSR matrix.
     */
    ~csr_matrix();

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `csr_matrix` to another to avoid shallow copies
     * and ensure proper memory management. Use `copy_from` for explicit copying.
     */
    csr_matrix& operator= (const csr_matrix&) = delete;

    /*! \brief Checks if the matrix data is currently stored on the host (CPU).
     * \return `true` if the matrix data is on the host, `false` otherwise (e.g., on a device).
     */
    bool is_on_host() const;

    /*! \brief Returns the number of rows in the matrix.
     * \return The number of rows (`m`).
     */
    int get_m() const;

    /*! \brief Returns the number of columns in the matrix.
     * \return The number of columns (`n`).
     */
    int get_n() const;

    /*! \brief Returns the number of non-zero elements in the matrix.
     * \return The number of non-zeros (`nnz`).
     */
    int get_nnz() const;

    /*! \brief Returns a constant pointer to the beginning of the row pointer array.
     * \return A `const int*` to `hcsr_row_ptr`.
     */
    const int* get_row_ptr() const;

    /*! \brief Returns a constant pointer to the beginning of the column indices array.
     * \return A `const int*` to `hcsr_col_ind`.
     */
    const int* get_col_ind() const;

    /*! \brief Returns a constant pointer to the beginning of the values array.
     * \return A `const double*` to `hcsr_val`.
     */
    const double* get_val() const;

    /*! \brief Returns a non-constant pointer to the beginning of the row pointer array.
     * \return An `int*` to `hcsr_row_ptr`. This allows modification of the array.
     */
    int* get_row_ptr();

    /*! \brief Returns a non-constant pointer to the beginning of the column indices array.
     * \return An `int*` to `hcsr_col_ind`. This allows modification of the array.
     */
    int* get_col_ind();

    /*! \brief Returns a non-constant pointer to the beginning of the values array.
     * \return A `double*` to `hcsr_val`. This allows modification of the array.
     */
    double* get_val();

    /*! \brief Resizes the CSR matrix and reallocates memory for its internal arrays.
     *
     * This method clears the existing data and sets the new dimensions and non-zero count.
     * All internal `std::vector`s are resized accordingly.
     * \param m The new number of rows.
     * \param n The new number of columns.
     * \param nnz The new number of non-zero elements.
     */
    void resize(int m, int n, int nnz);

    /*! \brief Copies the contents of another `csr_matrix` into this object.
     *
     * This performs a deep copy, ensuring that all data (row pointers, column indices,
     * and values) are duplicated.
     * \param A The source `csr_matrix` to copy from.
     */
    void copy_from(const csr_matrix& A);

    /*! \brief Extracts the diagonal elements of the matrix.
     *
     * This method populates the provided `vector` with the diagonal elements of the CSR matrix.
     * \param diag An output `vector` that will store the diagonal elements.
     */
    void extract_diagonal(vector& diag) const;

    /*! \brief Multiplies the CSR matrix by a vector: \f$y = A \cdot x\f$.
     *
     * Performs a sparse matrix-vector multiplication.
     * \param y The output `vector` to store the result of the multiplication.
     * \param x The input `vector` to multiply with the matrix.
     */
    void multiply_by_vector(vector& y, const vector& x) const;

    /*! \brief Multiplies the CSR matrix by a vector and adds the result: \f$y = y + A \cdot x\f$.
     *
     * Performs a sparse matrix-vector multiplication and accumulates the result into `y`.
     * \param y The input/output `vector`. On input, it contains initial values; on output,
     * it contains the accumulated result.
     * \param x The input `vector` to multiply with the matrix.
     */
    void multiply_by_vector_and_add(vector& y, const vector& x) const;

    /*! \brief Multiplies this CSR matrix by another CSR matrix: \f$C = A \cdot B\f$.
     *
     * Performs a sparse matrix-matrix multiplication where `this` is matrix A and `B` is matrix B.
     * \param C The output `csr_matrix` to store the product \f$A \cdot B\f$.
     * \param B The right-hand side `csr_matrix` in the multiplication.
     */
    void multiply_by_matrix(csr_matrix& C, const csr_matrix& B) const;

    /*! \brief Computes the transpose of the CSR matrix: \f$T = A^T\f$.
     *
     * \param T The output `csr_matrix` that will store the transpose of this matrix.
     */
    void transpose(csr_matrix& T) const;

    /*! \brief Moves the matrix data from host memory to device memory (e.g., GPU).
     * \details This method handles the necessary memory transfers if a device is available
     * and `on_host` is true. After this call, `is_on_host()` will return `false`.
     */
    void move_to_device();

    /*! \brief Moves the matrix data from device memory to host memory (e.g., CPU).
     * \details This method handles the necessary memory transfers if data is on a device
     * and `on_host` is false. After this call, `is_on_host()` will return `true`.
     */
    void move_to_host();

    /*! \brief Reads a sparse matrix from a Matrix Market (.mtx) file into this CSR object.
     * \param filename The path to the .mtx file.
     * \return `true` if the file was successfully read and parsed, `false` otherwise.
     */
    bool read_mtx(const std::string& filename);

    /*! \brief Writes the CSR matrix data to a Matrix Market (.mtx) file.
     * \param filename The path to the .mtx file where the matrix will be written.
     * \return `true` if the matrix was successfully written, `false` otherwise.
     */
    bool write_mtx(const std::string& filename);

    /*! \brief Modifies the matrix to be diagonally dominant.
     * \details This method typically adds a small value to the diagonal elements
     * to ensure strict diagonal dominance, which can be beneficial for the
     * convergence of iterative solvers.
     */
    void make_diagonally_dominant();

    /*! \brief Prints the contents of the CSR matrix to the console.
     * \param name A string identifier for the matrix, used in the print output.
     */
    void print_matrix(const std::string name) const;
};
}

#endif
