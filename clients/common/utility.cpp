#include "utility.h"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <cstring>

struct col_val
{
    int col_ind;
    double val;
};

bool load_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr, std::vector<int> &csr_col_ind,
                   std::vector<double> &csr_val, int &m, int &n, int &nnz)
{
    std::cout << "filename: " << filename << std::endl;
    std::ifstream file;

    file.open(filename.c_str());

    std::vector<int> row_ind;
    std::vector<int> col_ind;
    std::vector<double> vals;

    m = 0;
    n = 0;
    nnz = 0;

    int index = 0;
    if (file.is_open())
    {

        std::string percent("%");
        std::string space(" ");
        std::string token;

        int currentLine = 0;

        // scan through file
        while (!file.eof())
        {
            std::string line;
            std::getline(file, line);

            if (currentLine > 0 && index == nnz)
            {
                break;
            }

            // parse the line
            if (line.substr(0, 1).compare(percent) != 0)
            {
                if (currentLine == 0)
                {
                    std::cout << "line: " << line << std::endl;
                    token = line.substr(0, line.find(space));
                    m = atoi(token.c_str());
                    line.erase(0, line.find(space) + space.length());
                    token = line.substr(0, line.find(space));
                    n = atoi(token.c_str());
                    line.erase(0, line.find(space) + space.length());
                    token = line.substr(0, line.find(space));
                    int lower_triangular_nnz = atoi(token.c_str());

                    nnz = 2 * (lower_triangular_nnz - m) + m;

                    row_ind.resize(nnz);
                    col_ind.resize(nnz);
                    vals.resize(nnz);

                    std::cout << "m: " << m << " n: " << n << " nnz: " << nnz << std::endl;
                }

                if (currentLine > 0)
                {
                    token = line.substr(0, line.find(space));
                    int r = atoi(token.c_str()) - 1;
                    line.erase(0, line.find(space) + space.length());
                    token = line.substr(0, line.find(space));
                    int c = atoi(token.c_str()) - 1;
                    double v = 1.0;
                    if (line.find(space) != std::string::npos) // some mtx files do not have any values. In these cases
                                                               // just use a value of 1
                    {
                        line.erase(0, line.find(space) + space.length());
                        token = line.substr(0, line.find(space));
                        v = strtod(token.c_str(), NULL);
                    }

                    row_ind[index] = r;
                    col_ind[index] = c;
                    vals[index] = v;

                    index++;

                    if (r != c)
                    {
                        row_ind[index] = c;
                        col_ind[index] = r;
                        vals[index] = v;

                        index++;
                    }
                }
                currentLine++;
            }
        }

        file.close();
    }
    else
    {
        std::cout << "Could not open file: " << filename << std::endl;
        return false;
    }

    // find number of entries in each row;
    csr_row_ptr.resize(m + 1, 0);
    for (size_t i = 0; i < row_ind.size(); i++)
    {
        csr_row_ptr[row_ind[i] + 1]++;
    }

    for (int i = 0; i < m; i++)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // std::cout << "csr_row_ptr" << std::endl;
    // for (size_t i = 0; i < csr_row_ptr.size(); i++)
    //{
    //	std::cout << csr_row_ptr[i] << " ";
    // }
    // std::cout << "" << std::endl;

    csr_col_ind.resize(nnz, -1);
    csr_val.resize(nnz, 0.0);

    for (int i = 0; i < nnz; i++)
    {
        int row_start = csr_row_ptr[row_ind[i]];
        int row_end = csr_row_ptr[row_ind[i] + 1];

        for (int j = row_start; j < row_end; j++)
        {
            if (csr_col_ind[j] == -1)
            {
                csr_col_ind[j] = col_ind[i];
                csr_val[j] = vals[i];
                break;
            }
        }
    }

    // Verify no negative 1 found in csr column indices array
    for (size_t i = 0; i < csr_col_ind.size(); i++)
    {
        if (csr_col_ind[i] == -1)
        {
            std::cout << "Error in csr_co_ind array. Negative 1 found" << std::endl;
            return false;
        }
    }

    // Sort columns and values
    for (int i = 0; i < m; i++)
    {
        int row_start = csr_row_ptr[row_ind[i]];
        int row_end = csr_row_ptr[row_ind[i] + 1];

        std::vector<col_val> unsorted_col_vals(row_end - row_start);
        for (int j = row_start; j < row_end; j++)
        {
            unsorted_col_vals[j - row_start].col_ind = csr_col_ind[j];
            unsorted_col_vals[j - row_start].val = csr_val[j];
        }

        std::sort(unsorted_col_vals.begin(), unsorted_col_vals.end(),
                  [&](col_val t1, col_val t2) { return t1.col_ind < t2.col_ind; });

        for (int j = row_start; j < row_end; j++)
        {
            csr_col_ind[j] = unsorted_col_vals[j - row_start].col_ind;
            csr_val[j] = unsorted_col_vals[j - row_start].val;
        }
    }

    // std::cout << "csr_col_ind" << std::endl;
    // for (size_t i = 0; i < csr_col_ind.size(); i++)
    //{
    //	std::cout << csr_col_ind[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // std::cout << "csr_val" << std::endl;
    // for (size_t i = 0; i < csr_val.size(); i++)
    //{
    //	std::cout << csr_val[i] << " ";
    // }
    // std::cout << "" << std::endl;

    return true;
}

bool load_diagonally_dominant_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr,
                                       std::vector<int> &csr_col_ind, std::vector<double> &csr_val, int &m, int &n,
                                       int &nnz)
{
    load_mtx_file(filename, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);

    assert(((int)csr_row_ptr.size() - 1) == m);
    assert(((int)csr_val.size()) == nnz);

    // Return early is matrix has no diagonal
    int diagonal_count = 0;
    for (int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        for (int j = start; j < end; j++)
        {
            if (csr_col_ind[j] == i)
            {
                diagonal_count++;
                break;
            }
        }
    }

    if (diagonal_count < m)
    {
        return false;
    }

    // Make matrix diagonally dominant so that convergence is guaranteed
    for (int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        double row_sum = 0;
        for (int j = start; j < end; j++)
        {
            if (csr_col_ind[j] != i)
            {
                row_sum += std::abs(csr_val[j]);
            }
        }

        for (int j = start; j < end; j++)
        {
            if (csr_col_ind[j] == i)
            {
                csr_val[j] = std::max(std::abs(csr_val[j]), 1.1 * row_sum);
                break;
            }
        }
    }

    return true;
}

static void transpose(const csr_matrix &nontransposed, csr_matrix &transposed)
{
    transposed.m = nontransposed.n;
    transposed.n = nontransposed.m;
    transposed.nnz = nontransposed.nnz;
    transposed.csr_row_ptr.resize(transposed.m + 1);
    transposed.csr_col_ind.resize(transposed.nnz);
    transposed.csr_val.resize(transposed.nnz);

    // Fill arrays
    for (size_t i = 0; i < transposed.csr_row_ptr.size(); i++)
    {
        transposed.csr_row_ptr[i] = 0;
    }

    for (size_t i = 0; i < transposed.csr_col_ind.size(); i++)
    {
        transposed.csr_col_ind[i] = -1;
    }

    for (int i = 0; i < nontransposed.m; i++)
    {
        int row_start = nontransposed.csr_row_ptr[i];
        int row_end = nontransposed.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            transposed.csr_row_ptr[nontransposed.csr_col_ind[j] + 1]++;
        }
    }

    // Exclusive scan on row pointer array
    for (int i = 0; i < transposed.m; i++)
    {
        transposed.csr_row_ptr[i + 1] += transposed.csr_row_ptr[i];
    }

    for (int i = 0; i < nontransposed.m; i++)
    {
        int row_start = nontransposed.csr_row_ptr[i];
        int row_end = nontransposed.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            int col = nontransposed.csr_col_ind[j];
            double val = nontransposed.csr_val[j];

            int start = transposed.csr_row_ptr[col];
            int end = transposed.csr_row_ptr[col + 1];

            for (int k = start; k < end; k++)
            {
                if (transposed.csr_col_ind[k] == -1)
                {
                    transposed.csr_col_ind[k] = i;
                    transposed.csr_val[k] = val;
                    break;
                }
            }
        }
    }
}

// Compute C = alpha * A * B + beta * D
static void csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                        const int *csr_col_ind_A, const int *csr_row_ptr_B, const int *csr_col_ind_B, double beta,
                        const int *csr_row_ptr_D, const int *csr_col_ind_D, int *csr_row_ptr_C, int *nnz_C)
{
    std::vector<int> nnz(n, -1);

    // A is mxk, B is kxn, and C is mxn
    for (int i = 0; i < m + 1; i++)
    {
        csr_row_ptr_C[i] = 0;
    }

    for (int i = 0; i < m; ++i)
    {
        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            int row_begin_B = csr_row_ptr_B[col_A];
            int row_end_B = csr_row_ptr_B[col_A + 1];

            for (int p = row_begin_B; p < row_end_B; p++)
            {
                int col_B = csr_col_ind_B[p];

                if (nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    csr_row_ptr_C[i + 1]++;
                }
            }
        }

        if (beta != 0.0)
        {
            int row_begin_D = csr_row_ptr_D[i];
            int row_end_D = csr_row_ptr_D[i + 1];

            for (int j = row_begin_D; j < row_end_D; j++)
            {
                int col_D = csr_col_ind_D[j];

                if (nnz[col_D] != i)
                {
                    nnz[col_D] = i;
                    csr_row_ptr_C[i + 1]++;
                }
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    *nnz_C = csr_row_ptr_C[m];
}

static void csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                    const int *csr_col_ind_A, const double *csr_val_A, const int *csr_row_ptr_B,
                    const int *csr_col_ind_B, const double *csr_val_B, double beta, const int *csr_row_ptr_D,
                    const int *csr_col_ind_D, const double *csr_val_D, const int *csr_row_ptr_C, int *csr_col_ind_C,
                    double *csr_val_C)
{
    std::vector<int> nnzs(n, -1);

    for (int i = 0; i < m; i++)
    {
        int row_begin_C = csr_row_ptr_C[i];
        int row_end_C = row_begin_C;

        int row_begin_A = csr_row_ptr_A[i];
        int row_end_A = csr_row_ptr_A[i + 1];

        for (int j = row_begin_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];
            double val_A = alpha * csr_val_A[j];

            int row_begin_B = csr_row_ptr_B[col_A];
            int row_end_B = csr_row_ptr_B[col_A + 1];

            for (int p = row_begin_B; p < row_end_B; p++)
            {
                int col_B = csr_col_ind_B[p];
                double val_B = csr_val_B[p];

                if (nnzs[col_B] < row_begin_C)
                {
                    nnzs[col_B] = row_end_C;
                    csr_col_ind_C[row_end_C] = col_B;
                    csr_val_C[row_end_C] = val_A * val_B;
                    row_end_C++;
                }
                else
                {
                    csr_val_C[nnzs[col_B]] += val_A * val_B;
                }
            }
        }

        if (beta != 0.0)
        {
            int row_begin_D = csr_row_ptr_D[i];
            int row_end_D = csr_row_ptr_D[i + 1];

            for (int j = row_begin_D; j < row_end_D; j++)
            {
                int col_D = csr_col_ind_D[j];
                double val_D = beta * csr_val_D[j];

                // Check if a new nnz is generated or if the value is added
                if (nnzs[col_D] < row_begin_C)
                {
                    nnzs[col_D] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_D;
                    csr_val_C[row_end_C] = val_D;
                    row_end_C++;
                }
                else
                {
                    csr_val_C[nnzs[col_D]] += val_D;
                }
            }
        }
    }

    int nnz = csr_row_ptr_C[m];

    std::vector<int> cols(nnz);
    std::vector<double> vals(nnz);

    memcpy(cols.data(), csr_col_ind_C, sizeof(int) * nnz);
    memcpy(vals.data(), csr_val_C, sizeof(double) * nnz);

    for (int i = 0; i < m; i++)
    {
        int row_begin = csr_row_ptr_C[i];
        int row_end = csr_row_ptr_C[i + 1];
        int row_nnz = row_end - row_begin;

        std::vector<int> perm(row_nnz);
        for (int j = 0; j < row_nnz; j++)
        {
            perm[j] = j;
        }

        int *col_entry = cols.data() + row_begin;
        double *val_entry = vals.data() + row_begin;

        std::sort(perm.begin(), perm.end(), [&](const int &a, const int &b) { return col_entry[a] < col_entry[b]; });

        for (int j = 0; j < row_nnz; j++)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j] = val_entry[perm[j]];
        }
    }
}

bool load_spd_mtx_file(const std::string &filename, std::vector<int> &csr_row_ptr, std::vector<int> &csr_col_ind,
                       std::vector<double> &csr_val, int &m, int &n, int &nnz)
{
    csr_matrix nontransposed;
    load_mtx_file(filename, nontransposed.csr_row_ptr, nontransposed.csr_col_ind, nontransposed.csr_val,
                  nontransposed.m, nontransposed.n, nontransposed.nnz);

    csr_matrix transposed;
    transpose(nontransposed, transposed);

    // A = nontransposed * transposed
    m = nontransposed.m;
    n = transposed.n;
    nnz = 0;
    csr_row_ptr.resize(m + 1, 0);

    csrgemm_nnz(nontransposed.m, transposed.n, nontransposed.n, nontransposed.nnz, transposed.nnz, 0, 1.0,
                nontransposed.csr_row_ptr.data(), nontransposed.csr_col_ind.data(), transposed.csr_row_ptr.data(),
                transposed.csr_col_ind.data(), 0, nullptr, nullptr, csr_row_ptr.data(), &nnz);

    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);

    csrgemm(nontransposed.m, transposed.n, nontransposed.n, nontransposed.nnz, transposed.nnz, 0, 1.0,
            nontransposed.csr_row_ptr.data(), nontransposed.csr_col_ind.data(), nontransposed.csr_val.data(),
            transposed.csr_row_ptr.data(), transposed.csr_col_ind.data(), transposed.csr_val.data(), 0, nullptr,
            nullptr, nullptr, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());

    // Make matrix diagonally dominant
    for (int i = 0; i < m; i++)
    {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];

        double row_sum = 0;
        for (int j = start; j < end; j++)
        {
            if (csr_col_ind[j] != i)
            {
                row_sum += std::abs(csr_val[j]);
            }
        }

        for (int j = start; j < end; j++)
        {
            if (csr_col_ind[j] == i)
            {
                csr_val[j] = std::max(std::abs(csr_val[j]), 1.1 * row_sum);
                break;
            }
        }
    }

    return true;
}

bool check_solution(const std::vector<int> &csr_row_ptr, const std::vector<int> &csr_col_ind,
                    const std::vector<double> &csr_val, int m, int n, int nnz, const std::vector<double> &b,
                    const std::vector<double> &x, double tol)
{
    std::vector<double> residual(m, 0.0);
    for (int i = 0; i < m; i++)
    {
        int row_begin = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];

        double sum = 0;
        for (int j = row_begin; j < row_end; j++)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
        }

        residual[i] = sum - b[i];
    }

    bool solution_valid = true;
    for (size_t i = 0; i < x.size(); i++)
    {
        if (std::isnan(x[i]) || std::isinf(x[i]))
        {
            solution_valid = false;
            break;
        }
    }

    double max_error = 0.0;
    for (size_t i = 0; i < residual.size(); i++)
    {
        max_error = std::max(max_error, std::abs(residual[i]));
    }

    std::cout << "max_error: " << max_error << std::endl;

    return (max_error < tol && solution_valid);
}