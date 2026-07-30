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

#ifndef HOST_MATH_H
#define HOST_MATH_H

#include <string>

#include "host_axpy.h"
#include "host_csr2csc.h"
#include "host_csrgeam.h"
#include "host_csrgemm.h"
#include "host_csric0.h"
#include "host_csrilu0.h"
#include "host_csrtrsv.h"
#include "host_extract.h"
#include "host_matrix_vector.h"
#include "host_ruiz_scaling.h"
#include "host_scale.h"
#include "host_ssor.h"
#include "host_tridiagonal.h"

#include "linalg_export.h"

namespace linalg
{
    double host_norm_euclid(const vector<double>& array);
    double host_norm_inf(const vector<double>& array);
    void
        host_jacobi_solve(const vector<double>& rhs, const vector<double>& diag, vector<double>& x);
}

#endif
