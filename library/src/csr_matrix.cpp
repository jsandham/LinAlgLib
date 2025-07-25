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

#include "../include/csr_matrix.h"
#include "../include/linalg_math.h"

#include "trace.h"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace linalg;

csr_matrix::csr_matrix()
    : m(0)
    , n(0)
    , nnz(0)
    , on_host(true)
{
}

csr_matrix::csr_matrix(const std::vector<int>&    csr_row_ptr,
                       const std::vector<int>&    csr_col_ind,
                       const std::vector<double>& csr_val,
                       int                        m,
                       int                        n,
                       int                        nnz)
{
    this->csr_row_ptr.resize(csr_row_ptr.size());
    this->csr_col_ind.resize(csr_col_ind.size());
    this->csr_val.resize(csr_val.size());

    this->csr_row_ptr.copy_from(csr_row_ptr); // = csr_row_ptr;
    this->csr_col_ind.copy_from(csr_col_ind); // = csr_col_ind;
    this->csr_val.copy_from(csr_val); // = csr_val;
    this->m   = m;
    this->n   = n;
    this->nnz = nnz;

    this->on_host = true;
}
csr_matrix::~csr_matrix() {}

bool csr_matrix::is_on_host() const
{
    return on_host;
}

int csr_matrix::get_m() const
{
    return this->m;
}

int csr_matrix::get_n() const
{
    return this->n;
}

int csr_matrix::get_nnz() const
{
    return this->nnz;
}

const int* csr_matrix::get_row_ptr() const
{
    return csr_row_ptr.get_vec();
}

const int* csr_matrix::get_col_ind() const
{
    return csr_col_ind.get_vec();
}

const double* csr_matrix::get_val() const
{
    return csr_val.get_vec();
}

int* csr_matrix::get_row_ptr()
{
    return csr_row_ptr.get_vec();
}

int* csr_matrix::get_col_ind()
{
    return csr_col_ind.get_vec();
}

double* csr_matrix::get_val()
{
    return csr_val.get_vec();
}

void csr_matrix::resize(int m, int n, int nnz)
{
    ROUTINE_TRACE("csr_matrix::resize");
    this->csr_row_ptr.resize(m + 1);
    this->csr_col_ind.resize(nnz);
    this->csr_val.resize(nnz);
    this->m   = m;
    this->n   = n;
    this->nnz = nnz;
}

void csr_matrix::copy_from(const csr_matrix& A)
{
    ROUTINE_TRACE("csr_matrix::copy_from");

    this->m   = A.get_m();
    this->n   = A.get_n();
    this->nnz = A.get_nnz();
    this->csr_row_ptr.resize(A.get_m() + 1);
    this->csr_col_ind.resize(A.get_nnz());
    this->csr_val.resize(A.get_nnz());

    this->csr_row_ptr.copy_from(A.csr_row_ptr);
    this->csr_col_ind.copy_from(A.csr_col_ind);
    this->csr_val.copy_from(A.csr_val);
}

void csr_matrix::move_to_device()
{
    ROUTINE_TRACE("csr_matrix::move_to_device");

    // if(!is_device_available())
    // {
    //     std::cout << "Warning: Device not available. Keeping matrix on the host." << std::endl;
    //     return;
    // }

    csr_row_ptr.move_to_device();
    csr_col_ind.move_to_device();
    csr_val.move_to_device();
}

void csr_matrix::move_to_host()
{
    ROUTINE_TRACE("csr_matrix::move_to_host");

    csr_row_ptr.move_to_host();
    csr_col_ind.move_to_host();
    csr_val.move_to_host();
}

void csr_matrix::extract_diagonal(vector<double>& diag) const
{
    ROUTINE_TRACE("csr_matrix::extract_diagonal");

    diagonal(*this, diag);
}

void csr_matrix::multiply_by_vector(vector<double>& y, const vector<double>& x) const
{
    ROUTINE_TRACE("csr_matrix::multiply_by_vector");

    matrix_vector_product(*this, x, y);
}

void csr_matrix::multiply_by_vector_and_add(vector<double>& y, const vector<double>& x) const
{
    ROUTINE_TRACE("csr_matrix::multiply_by_vector_and_add");

    matrix_vector_product(1.0, *this, x, 1.0, y);
}

void csr_matrix::multiply_by_matrix(csr_matrix& C, const csr_matrix& B) const
{
    ROUTINE_TRACE("csr_matrix::multiply_by_matrix");

    matrix_matrix_product(C, *this, B);
}

void csr_matrix::transpose(csr_matrix& T) const
{
    ROUTINE_TRACE("csr_matrix::transpose_matrix");

    transpose_matrix(*this, T);
}

// Structure to hold triplet (COO) format data
struct triplet
{
    int    row;
    int    col;
    double value;

    // For sorting: primarily by row, then by column
    bool operator<(const triplet& other) const
    {
        if(row != other.row)
        {
            return row < other.row;
        }
        return col < other.col;
    }
};

bool csr_matrix::read_mtx(const std::string& filename)
{
    ROUTINE_TRACE("csr_matrix::read_mtx");

    if(!this->is_on_host())
    {
        std::cout << "Matrix must be on the host in order to read from matrix market file"
                  << std::endl;
        return false;
    }

    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string header;
    bool        is_symmetric = false;
    bool        is_integer   = false;

    // Read header line
    std::getline(file, header);
    std::stringstream header_ss(header);
    std::string       token;
    header_ss >> token; // %%MatrixMarket

    // Check object type (matrix)
    header_ss >> token;
    if(token != "matrix")
    {
        std::cerr << "Error: Not a matrix market file." << std::endl;
        return false;
    }

    // Check format (array or coordinate)
    header_ss >> token;
    if(token != "coordinate")
    {
        std::cerr << "Error: Only 'coordinate' format is supported (not 'array')." << std::endl;
        return false;
    }

    // Check data type (real, integer, complex, pattern)
    header_ss >> token;
    if(token == "real")
    {
        is_integer = false;
    }
    else if(token == "integer")
    {
        is_integer = true;
    }
    else if(token == "complex" || token == "pattern")
    {
        std::cerr << "Error: 'complex' and 'pattern' data types are not supported for values."
                  << std::endl;
        return false;
    }
    else
    {
        std::cerr << "Error: Unknown data type in Matrix Market header: " << token << std::endl;
        return false;
    }

    // Check symmetry (general, symmetric, Hermitian, skew-symmetric)
    header_ss >> token;
    if(token == "general")
    {
        is_symmetric = false;
    }
    else if(token == "symmetric")
    {
        is_symmetric = true;
    }
    else if(token == "hermitian" || token == "skew-symmetric")
    {
        std::cerr << "Error: 'Hermitian' and 'skew-symmetric' matrices are not supported."
                  << std::endl;
        return false;
    }
    else
    {
        std::cerr << "Error: Unknown symmetry type in Matrix Market header: " << token << std::endl;
        return false;
    }

    // Skip comment lines
    while(std::getline(file, line) && line[0] == '%')
    {
        // Do nothing, just consume comments
    }

    // Read dimensions and number of non-zero elements
    std::stringstream ss(line);
    int64_t           nnz_coo; // Non-zeros as stated in the file (COO count)
    if(!(ss >> m >> n >> nnz_coo))
    {
        std::cerr << "Error: Failed to read matrix dimensions or number of non-zeros." << std::endl;
        return false;
    }

    std::vector<triplet> triplets;
    triplets.reserve(is_symmetric ? nnz_coo * 2
                                  : nnz_coo); // Reserve enough space for symmetric case

    // Read non-zero elements
    int     r, c;
    double  val_double;
    int64_t val_long;

    for(int64_t i = 0; i < nnz_coo; ++i)
    {
        if(!(std::getline(file, line)))
        {
            std::cerr << "Error: Unexpected end of file while reading elements (expected "
                      << nnz_coo << " elements)." << std::endl;
            return false;
        }
        std::stringstream element_ss(line);

        if(is_integer)
        {
            if(!(element_ss >> r >> c >> val_long))
            {
                std::cerr << "Error: Failed to parse integer element at line " << i + 1 << "."
                          << std::endl;
                return false;
            }
            triplets.push_back(
                {r - 1, c - 1, static_cast<double>(val_long)}); // Matrix Market is 1-indexed
        }
        else // real
        {
            if(!(element_ss >> r >> c >> val_double))
            {
                std::cerr << "Error: Failed to parse real element at line " << i + 1 << "."
                          << std::endl;
                return false;
            }
            triplets.push_back({r - 1, c - 1, val_double}); // Matrix Market is 1-indexed
        }

        // Handle symmetric case: add the (c, r) entry if r != c
        if(is_symmetric && r - 1 != c - 1)
        {
            triplets.push_back({c - 1, r - 1, triplets.back().value});
        }
    }

    file.close();

    // Sort triplets to group by row, then by column. This is crucial for CSR.
    std::sort(triplets.begin(), triplets.end());

    // Remove duplicate entries (e.g., if a symmetric matrix had (i,j) and (j,i) explicitly listed,
    // or if (i,i) was explicitly listed and then added again by symmetric handling).
    // Also, sum values if multiple entries refer to the same (row, col)
    if(!triplets.empty())
    {
        std::vector<triplet> unique_triplets;
        unique_triplets.reserve(triplets.size());
        unique_triplets.push_back(triplets[0]);

        for(size_t i = 1; i < triplets.size(); ++i)
        {
            if(triplets[i].row == unique_triplets.back().row
               && triplets[i].col == unique_triplets.back().col)
            {
                // If duplicate, sum values
                unique_triplets.back().value += triplets[i].value;
            }
            else
            {
                unique_triplets.push_back(triplets[i]);
            }
        }
        triplets = std::move(unique_triplets); // Use move assignment
    }

    // Convert to CSR format
    nnz = triplets.size();

    if(nnz == 0)
    {
        // Handle empty matrix case: all dimensions valid but no entries
        // csr_row_ptr.assign(m + 1, 0);
        csr_row_ptr.resize(m + 1);
        csr_row_ptr.zeros();
        csr_col_ind.clear();
        csr_val.clear();
        return true;
    }

    // csr_row_ptr.assign(m + 1, 0); // Initialize with zeros
    csr_row_ptr.resize(m + 1);
    csr_row_ptr.zeros(); // Initialize with zeros
    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    for(int64_t i = 0; i < nnz; ++i)
    {
        const triplet& t = triplets[i];

        csr_row_ptr[t.row + 1]++;
        csr_col_ind[i] = t.col;
        csr_val[i]     = t.value;
    }

    // Convert counts to cumulative sum (prefix sum) for row_ptr
    for(int i = 0; i < m; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    return true;
}

bool csr_matrix::write_mtx(const std::string& filename)
{
    ROUTINE_TRACE("csr_matrix::write_mtx");

    if(!this->is_on_host())
    {
        std::cout << "Matrix must be on the host in order to write to matrix market file"
                  << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    // Optimization: Faster I/O for large files
    // Note: This affects only the output stream 'file', not global std::cout
    file.sync_with_stdio(false);
    file.tie(NULL);

    // Determine data type for the header
    std::string data_type_str;
    if(std::is_floating_point<double>::value)
    {
        data_type_str = "real";
    }
    else if(std::is_integral<double>::value)
    {
        data_type_str = "integer";
    }
    else
    {
        std::cerr << "Error: Unsupported data type for Matrix Market file." << std::endl;
        return false;
    }

    // Write Matrix Market header
    file << "%%MatrixMarket matrix coordinate " << data_type_str << " general\n";
    file << "% Created by C++ CSR to Matrix Market writer\n";
    file << m << " " << n << " " << nnz << "\n";

    // Set precision for floating-point numbers if applicable
    if(std::is_floating_point<double>::value)
    {
        file << std::fixed << std::setprecision(10); // Adjust precision as needed
    }

    // Write non-zero elements
    for(int i = 0; i < m; ++i)
    {
        for(int j_idx = csr_row_ptr[i]; j_idx < csr_row_ptr[i + 1]; ++j_idx)
        {
            // Matrix Market uses 1-based indexing
            file << (i + 1) << " " << (csr_col_ind[j_idx] + 1) << " " << csr_val[j_idx] << "\n";
        }
    }

    file.close();
    return true;
}

void csr_matrix::make_diagonally_dominant()
{
    ROUTINE_TRACE("csr_matrix::make_diagonally_dominant");

    if(!this->is_on_host())
    {
        std::cout << "Matrix must be on the host in order to make diagonally dominant" << std::endl;
        return;
    }

    assert(((int)csr_row_ptr.get_size() - 1) == m);
    assert(((int)csr_val.get_size()) == nnz);

    // Return early is matrix has no diagonal
    int diagonal_count = 0;
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end   = csr_row_ptr[i + 1];

        for(int j = start; j < end; j++)
        {
            if(csr_col_ind[j] == i)
            {
                diagonal_count++;
                break;
            }
        }
    }

    if(diagonal_count < m)
    {
        // Error?
    }

    // Make matrix diagonally dominant so that convergence is guaranteed
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end   = csr_row_ptr[i + 1];

        double row_sum = 0;
        for(int j = start; j < end; j++)
        {
            if(csr_col_ind[j] != i)
            {
                row_sum += std::abs(csr_val[j]);
            }
        }

        for(int j = start; j < end; j++)
        {
            if(csr_col_ind[j] == i)
            {
                csr_val[j] = std::max(std::abs(csr_val[j]), 1.1 * row_sum);
                break;
            }
        }
    }
}

void csr_matrix::print_matrix(const std::string name) const
{
    ROUTINE_TRACE("csr_matrix::print_matrix");

    if(!this->is_on_host())
    {
        std::cout << "Matrix must be on the host in order to print to console" << std::endl;
        return;
    }

    std::cout << name << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end   = csr_row_ptr[i + 1];

        std::vector<double> temp(n, 0.0);
        for(int j = start; j < end; j++)
        {
            temp[csr_col_ind[j]] = (csr_val.get_size() != 0) ? csr_val[j] : 1.0;
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;
}
