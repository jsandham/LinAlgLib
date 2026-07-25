//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#include <algorithm>
#include <assert.h>
#include <iostream>

#include "host_ruiz_scaling.h"

#include "../../trace.h"

namespace linalg
{
    template <typename T>
    static void host_ruiz_scaling_impl(
        T* D1, const int* csr_row_ptr, const int* csr_col_ind, T* csr_val, int m, T* D2)
    {
        ROUTINE_TRACE("host_scale_diagonal_impl");

        int k = 10;

        // D_1^0 = I, D_2^0 = I
        for(int i = 0; i < m; i++)
        {
            D1[i] = static_cast<T>(1);
            D2[i] = static_cast<T>(1);
        }

        std::vector<T> DR(m, static_cast<T>(0));
        std::vector<T> DC(m, static_cast<T>(0));

        for(int iter = 0; iter < k; iter++)
        {
            for(int i = 0; i < m; i++)
            {
                const int start = csr_row_ptr[i];
                const int end   = csr_row_ptr[i + 1];

                for(int j = start; j < end; j++)
                {
                    const int col = csr_col_ind[j];
                    const T   val = csr_val[j];

                    DR[i]   = std::max(DR[i], std::abs(val));
                    DC[col] = std::max(DC[col], std::abs(val));
                }
            }

            T row_max = static_cast<T>(0);
            T col_max = static_cast<T>(0);
            for(int i = 0; i < m; i++)
            {
                row_max = std::max(row_max, DR[i]);
                col_max = std::max(col_max, DC[i]);

                DR[i] = std::sqrt(DR[i]);
                DC[i] = std::sqrt(DC[i]);
            }

            std::cout << "row_max: " << row_max << " col_max: " << col_max << std::endl;

            // Update A_k+1 = DR^-1 * A_k * DC^-1
            for(int i = 0; i < m; i++)
            {
                const int start = csr_row_ptr[i];
                const int end   = csr_row_ptr[i + 1];

                for(int j = start; j < end; j++)
                {
                    const int col = csr_col_ind[j];
                    csr_val[j] /= (DR[i] * DC[col]);
                }
            }

            // Update D1_k+1 = D1_k * DR^-1, D2_k+1 = D2_k * DC^-1
            for(int i = 0; i < m; i++)
            {
                D1[i] = D1[i] / DR[i];
                D2[i] = D2[i] / DC[i];
            }
        }
    }
}

void linalg::host_ruiz_scaling(vector<double>& D1, csr_matrix& A, vector<double>& D2)
{
    ROUTINE_TRACE("linalg::host_ruiz_scaling");

    assert(A.get_m() == A.get_n());
    assert(D1.get_size() == A.get_m());
    assert(D2.get_size() == A.get_m());

    host_ruiz_scaling_impl(
        D1.get_vec(), A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), D2.get_vec());
}
