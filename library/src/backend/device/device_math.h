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

#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include <string>

#include "device_axpy.h"
#include "device_csr2csc.h"
#include "device_csrgeam.h"
#include "device_csrgemm.h"
#include "device_csric0.h"
#include "device_csrilu0.h"
#include "device_csrtrsv.h"
#include "device_extract.h"
#include "device_matrix_vector.h"
#include "device_scale.h"
#include "device_ssor.h"
#include "device_tridiagonal.h"

#include "linalg_export.h"

/*! \file
 *  \brief device_math.h provides linear algebra APIs for device (GPU) backend
 */
namespace linalg
{
    // Euclidean norm
    double device_norm_euclid(const vector<double>& array);

    // Infinity norm
    double device_norm_inf(const vector<double>& array);

    // Jacobi solve
    void device_jacobi_solve(const vector<double>& rhs,
                             const vector<double>& diag,
                             vector<double>&       x);

} // namespace linalg

#endif
