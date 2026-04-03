//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#include "host_extract.h"

#include "../../trace.h"

namespace linalg
{
    template <typename T>
    static void host_diagonal_impl(
        const int* csr_row_ptr, const int* csr_col_ind, const T* csr_val, T* d, int n)
    {
        ROUTINE_TRACE("host_diagonal_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < n; i++)
        {
            for(int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
            {
                if(csr_col_ind[j] == i)
                {
                    d[i] = csr_val[j];
                    break;
                }
            }
        }
    }

    static void host_extract_lower_triangular_nnz_impl(int        m_A,
                                                       int        n_A,
                                                       int        nnz_A,
                                                       const int* csr_row_ptr_A,
                                                       const int* csr_col_ind_A,
                                                       int*       csr_row_ptr_L,
                                                       int*       nnz_L)
    {
        ROUTINE_TRACE("host_extract_lower_triangular_nnz_impl");
        csr_row_ptr_L[0] = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int nnz_per_row = 0;
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                if(csr_col_ind_A[j] <= i)
                {
                    nnz_per_row++;
                }
            }
            csr_row_ptr_L[i + 1] = nnz_per_row;
        }
        for(int i = 0; i < m_A; i++)
        {
            csr_row_ptr_L[i + 1] += csr_row_ptr_L[i];
        }
        *nnz_L = csr_row_ptr_L[m_A];
    }

    template <typename T>
    static void host_extract_lower_triangular_impl(int        m_A,
                                                   int        n_A,
                                                   int        nnz_A,
                                                   const int* csr_row_ptr_A,
                                                   const int* csr_col_ind_A,
                                                   const T*   csr_val_A,
                                                   int        m_L,
                                                   int        n_L,
                                                   int        nnz_L,
                                                   int*       csr_row_ptr_L,
                                                   int*       csr_col_ind_L,
                                                   T*         csr_val_L)
    {
        ROUTINE_TRACE("host_extract_lower_triangular_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int index = csr_row_ptr_L[i];
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                if(csr_col_ind_A[j] <= i)
                {
                    csr_col_ind_L[index] = csr_col_ind_A[j];
                    csr_val_L[index]     = csr_val_A[j];
                    index++;
                }
            }
        }
    }

    static void host_extract_upper_triangular_nnz_impl(int        m_A,
                                                       int        n_A,
                                                       int        nnz_A,
                                                       const int* csr_row_ptr_A,
                                                       const int* csr_col_ind_A,
                                                       int*       csr_row_ptr_U,
                                                       int*       nnz_U)
    {
        ROUTINE_TRACE("host_extract_upper_triangular_nnz_impl");
        csr_row_ptr_U[0] = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int nnz_per_row = 0;
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                if(csr_col_ind_A[j] >= i)
                {
                    nnz_per_row++;
                }
            }
            csr_row_ptr_U[i + 1] = nnz_per_row;
        }
        for(int i = 0; i < m_A; i++)
        {
            csr_row_ptr_U[i + 1] += csr_row_ptr_U[i];
        }
        *nnz_U = csr_row_ptr_U[m_A];
    }

    template <typename T>
    static void host_extract_upper_triangular_impl(int        m_A,
                                                   int        n_A,
                                                   int        nnz_A,
                                                   const int* csr_row_ptr_A,
                                                   const int* csr_col_ind_A,
                                                   const T*   csr_val_A,
                                                   int        m_U,
                                                   int        n_U,
                                                   int        nnz_U,
                                                   int*       csr_row_ptr_U,
                                                   int*       csr_col_ind_U,
                                                   T*         csr_val_U)
    {
        ROUTINE_TRACE("host_extract_upper_triangular_impl");
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < m_A; i++)
        {
            int index = csr_row_ptr_U[i];
            for(int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            {
                if(csr_col_ind_A[j] >= i)
                {
                    csr_col_ind_U[index] = csr_col_ind_A[j];
                    csr_val_U[index]     = csr_val_A[j];
                    index++;
                }
            }
        }
    }
}

void linalg::host_diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::host_diagonal");
    host_diagonal_impl(A.get_row_ptr(), A.get_col_ind(), A.get_val(), d.get_vec(), A.get_m());
}

void linalg::host_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L)
{
    ROUTINE_TRACE("linalg::host_extract_lower_triangular_nnz");
    host_extract_lower_triangular_nnz_impl(A.get_m(),
                                           A.get_n(),
                                           A.get_nnz(),
                                           A.get_row_ptr(),
                                           A.get_col_ind(),
                                           L.get_row_ptr(),
                                           &nnz_L);
}

void linalg::host_extract_lower_triangular(const csr_matrix& A, csr_matrix& L)
{
    ROUTINE_TRACE("linalg::host_extract_lower_triangular");
    host_extract_lower_triangular_impl(A.get_m(),
                                       A.get_n(),
                                       A.get_nnz(),
                                       A.get_row_ptr(),
                                       A.get_col_ind(),
                                       A.get_val(),
                                       L.get_m(),
                                       L.get_n(),
                                       L.get_nnz(),
                                       L.get_row_ptr(),
                                       L.get_col_ind(),
                                       L.get_val());
}

void linalg::host_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U)
{
    ROUTINE_TRACE("linalg::host_extract_upper_triangular_nnz");
    host_extract_upper_triangular_nnz_impl(A.get_m(),
                                           A.get_n(),
                                           A.get_nnz(),
                                           A.get_row_ptr(),
                                           A.get_col_ind(),
                                           U.get_row_ptr(),
                                           &nnz_U);
}

void linalg::host_extract_upper_triangular(const csr_matrix& A, csr_matrix& U)
{
    ROUTINE_TRACE("linalg::host_extract_upper_triangular");
    host_extract_upper_triangular_impl(A.get_m(),
                                       A.get_n(),
                                       A.get_nnz(),
                                       A.get_row_ptr(),
                                       A.get_col_ind(),
                                       A.get_val(),
                                       U.get_m(),
                                       U.get_n(),
                                       U.get_nnz(),
                                       U.get_row_ptr(),
                                       U.get_col_ind(),
                                       U.get_val());
}
