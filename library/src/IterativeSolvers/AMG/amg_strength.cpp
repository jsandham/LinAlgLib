//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include "../../../include/IterativeSolvers/AMG/amg_strength.h"
#include "../../../include/slaf.h"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../../trace.h"

void compute_strong_connections(const csr_matrix &A, double eps, std::vector<int> &connections)
{
    ROUTINE_TRACE("compute_strong_connections");

    // Extract diagaonl
    std::vector<double> diag(A.m);
    diagonal(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), diag.data(), A.m);

    // double eps2 = eps * eps;

    for (int i = 0; i < A.m; i++)
    {
        // double eps_dia_i = eps2 * diag[i];

        int row_start = A.csr_row_ptr[i];
        int row_end = A.csr_row_ptr[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            int c = A.csr_col_ind[j];
            double v = A.csr_val[j];

            assert(c >= 0);
            assert(c < A.m);

            // connections[j] = (c != i) && (v * v > eps_dia_i * diag[c]);
            connections[j] = (c != i) && (std::abs(v) >= eps * std::sqrt(std::abs(diag[i]) * std::abs(diag[c])));
        }
    }
}

void compute_strong_connections(const csr_matrix2 &A, double eps, std::vector<int> &connections)
{
    ROUTINE_TRACE("compute_strong_connections");

    const int* csr_row_ptr_A = A.get_row_ptr();
    const int* csr_col_ind_A = A.get_col_ind();
    const double* csr_val_A = A.get_val();

    // Extract diagaonl
    std::vector<double> diag(A.get_m());
    diagonal(csr_row_ptr_A, csr_col_ind_A, csr_val_A, diag.data(), A.get_m());

    // double eps2 = eps * eps;

    for (int i = 0; i < A.get_m(); i++)
    {
        // double eps_dia_i = eps2 * diag[i];

        int row_start = csr_row_ptr_A[i];
        int row_end = csr_row_ptr_A[i + 1];

        for (int j = row_start; j < row_end; j++)
        {
            int c = csr_col_ind_A[j];
            double v = csr_val_A[j];

            assert(c >= 0);
            assert(c < A.get_m());

            // connections[j] = (c != i) && (v * v > eps_dia_i * diag[c]);
            connections[j] = (c != i) && (std::abs(v) >= eps * std::sqrt(std::abs(diag[i]) * std::abs(diag[c])));
        }
    }
}

//-------------------------------------------------------------------------------
// -A[i,j] >= theta * max( -A[i,k] )   where k != i
//-------------------------------------------------------------------------------
void compute_classical_strong_connections(const csr_matrix &A, double theta, csr_matrix &S, std::vector<int> &connections)
{
    S.m = A.m;
    S.n = A.n;
    S.csr_row_ptr.resize(S.m + 1);

    for(size_t i = 0; i < S.csr_row_ptr.size(); i++)
    {
        S.csr_row_ptr[i] = 0;
    }

    for(int i = 0; i < A.m; i++)
    {
        int row_start = A.csr_row_ptr[i];
        int row_end = A.csr_row_ptr[i + 1];

        double max_value = std::numeric_limits<double>::lowest(); // smallest, most negative, double
        for(int j = row_start; j < row_end; j++)
        {
            int col = A.csr_col_ind[j];
            double val = A.csr_val[j];

            if(i != col)
            {
                max_value = std::max(max_value, -val);
            }
        }

        // Fill connections array
        for(int j = row_start; j < row_end; j++)
        {
            int col = A.csr_col_ind[j];
            double val = A.csr_val[j];

            if(-val >= theta * max_value && i != col)
            {
                connections[j] = 1;
                S.csr_row_ptr[i + 1]++;
            }
        }
    }

    // Exclusive scan on S row pointer array
    for(int i = 0; i < S.m; i++)
    {
        S.csr_row_ptr[i + 1] += S.csr_row_ptr[i];
    }

    S.nnz = S.csr_row_ptr[S.m];
    S.csr_col_ind.resize(S.nnz);
    S.csr_val.resize(S.nnz);

    for(int i = 0; i < A.m; i++)
    {
        int row_start = A.csr_row_ptr[i];
        int row_end = A.csr_row_ptr[i + 1];

        int S_row_start = S.csr_row_ptr[i];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                S.csr_col_ind[S_row_start] = A.csr_col_ind[j];
                S.csr_val[S_row_start] = A.csr_val[j];
                S_row_start++;
            }
        }
    }
}

void compute_classical_strong_connections(const csr_matrix2 &A, double theta, csr_matrix2 &S, std::vector<int> &connections)
{
    //S.m = A.m;
    //S.n = A.n;
    //S.csr_row_ptr.resize(S.m + 1);
    S.resize(A.get_m(), A.get_n(), 0);

    int* csr_row_ptr_S = S.get_row_ptr();

    for(int i = 0; i < S.get_m() + 1; i++)
    {
        csr_row_ptr_S[i] = 0;
    }

    const int* csr_row_ptr_A = A.get_row_ptr();
    const int* csr_col_ind_A = A.get_col_ind();
    const double* csr_val_A = A.get_val();

    for(int i = 0; i < A.get_m(); i++)
    {
        int row_start = csr_row_ptr_A[i];
        int row_end = csr_row_ptr_A[i + 1];

        double max_value = std::numeric_limits<double>::lowest(); // smallest, most negative, double
        for(int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind_A[j];
            double val = csr_val_A[j];

            if(i != col)
            {
                max_value = std::max(max_value, -val);
            }
        }

        // Fill connections array
        for(int j = row_start; j < row_end; j++)
        {
            int col = csr_col_ind_A[j];
            double val = csr_val_A[j];

            if(-val >= theta * max_value && i != col)
            {
                connections[j] = 1;
                csr_row_ptr_S[i + 1]++;
            }
        }
    }

    // Exclusive scan on S row pointer array
    for(int i = 0; i < S.get_m(); i++)
    {
        csr_row_ptr_S[i + 1] += csr_row_ptr_S[i];
    }

    S.resize(S.get_m(), S.get_n(), csr_row_ptr_S[S.get_m()]);
    //S.nnz = S.csr_row_ptr[S.m];
    //S.csr_col_ind.resize(S.nnz);
    //S.csr_val.resize(S.nnz);

    int* csr_col_ind_S = S.get_col_ind();
    double* csr_val_S = S.get_val();

    for(int i = 0; i < A.get_m(); i++)
    {
        int row_start = csr_row_ptr_A[i];
        int row_end = csr_row_ptr_A[i + 1];

        int S_row_start = csr_row_ptr_S[i];

        for(int j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                csr_col_ind_S[S_row_start] = csr_col_ind_A[j];
                csr_val_S[S_row_start] = csr_val_A[j];
                S_row_start++;
            }
        }
    }
}
