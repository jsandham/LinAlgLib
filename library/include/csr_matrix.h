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

/*! \ingroup iterative_solvers
 * \brief Data class for storing sparse CSR matrices
 *
 * \details
 * This class represents a sparse matrix stored in the Compressed Sparse Row (CSR) format.
 * The CSR format is efficient for storing and operating on sparse matrices, as it only
 * stores the non-zero elements along with information about their row and column indices.
 */
class csr_matrix2
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

    bool on_host;

public:
    csr_matrix2();
    csr_matrix2(const std::vector<int>& csr_row_ptr, const std::vector<int>& csr_col_ind, const std::vector<double>& csr_val, int m, int n, int nnz);
    ~csr_matrix2();

    csr_matrix2 (const csr_matrix2&) = delete;
    csr_matrix2& operator= (const csr_matrix2&) = delete;

    bool is_on_host() const;

    int get_m() const;
    int get_n() const;
    int get_nnz() const;

    const int* get_row_ptr() const;
    const int* get_col_ind() const;
    const double* get_val() const;

    int* get_row_ptr();
    int* get_col_ind();
    double* get_val();

    void resize(int m, int n, int nnz);
    void copy_from(const csr_matrix2& A);

    void extract_diagonal(vector2& diag) const;
    void multiply_vector(vector2& y, const vector2& x) const;
    void multiply_vector_and_add(vector2& y, const vector2& x) const;
    void multiply_matrix(csr_matrix2& C, const csr_matrix2& B) const;
    void transpose(csr_matrix2& T) const;

    void move_to_device();
    void move_to_host();

    bool read_mtx(const std::string& filename);
    bool write_mtx(const std::string& filename);

    void make_diagonally_dominant();

    void print_matrix(const std::string name) const;
};

#endif
