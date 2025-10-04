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

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <string>

#include "linalg_export.h"
#include "vector.h"

/*! \file
 *  \brief linalg_primitives.h provides primitive operations like max/min, scans, sorting etc
 */

namespace linalg
{
    /**
     * @brief Finds the minimum value within a vector.
     *
     * This function iterates through the provided vector and returns the smallest
     * element found. It assumes the vector is not empty.
     *
     * @param x A constant reference to the input vector of ($\text{linalg::vector}< \text{double} >$)
     * to search for the minimum element.
     *
     * @return The minimum value found in the vector $\text{x}$.
     *
     * @see find_max
     *
     * @par Example
     * @code
     * std::vector<double> data = {5.1, -2.3, 9.0, 0.5, -4.2};
     * double minimum = find_min(data);
     * // minimum will be -4.2
     * @endcode
     */
    template<typename T>
    LINALGLIB_API T find_min(const vector<T>& x);

    /**
     * @brief Finds the maximum value within a vector.
     *
     * This function iterates through the provided vector and returns the largest
     * element found. It assumes the vector is not empty.
     *
     * @param x A constant reference to the input vector ($\text{linalg::vector}< \text{double} >$)
     * to search for the maximum element.
     *
     * @return The maximum value found in the vector $\text{x}$.
     *
     * @see find_min
     *
     * @par Example
     * @code
     * std::vector<double> data = {5.1, -2.3, 9.0, 0.5, -4.2};
     * double maximum = find_max(data);
     * // maximum will be 9.0
     * @endcode
     */
    template<typename T>
    LINALGLIB_API T find_max(const vector<T>& x);

    /**
     * @brief Performs an exclusive scan (prefix sum) on a vector.
     *
     * This function replaces each element with the sum of all elements preceding it.
     * The first element of the output vector will be 0.
     * For example, if input `x` is `{a, b, c}`, output `x` will be `{0, a, a+b}`.
     *
     * @param x The input/output vector. On input, it contains the original values; on output,
     * it contains the exclusive scan results.
     */
    template <typename T>
    LINALGLIB_API void exclusive_scan(vector<T>& x);
}

#endif