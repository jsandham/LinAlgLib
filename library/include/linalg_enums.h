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

#ifndef LINEAR_ENUMS_H
#define LINEAR_ENUMS_H

namespace linalg
{
    /*! \brief Enumeration for triangular matrix types. */
    enum class triangular_type
    {
        lower, /*!< Lower triangular matrix */
        upper /*!< Upper triangular matrix */
    };

    /*! \brief Enumeration for diagonal matrix types. */
    enum class diagonal_type
    {
        non_unit, /*!< Non-unit diagonal */
        unit /*!< Unit diagonal */
    };

    /*! \brief Enumeration for CSR matrix-vector multiplication algorithms. */
    enum class csrmv_algorithm
    {
        default_algorithm, /*!< Default algorithm */
        merge_path,
        rowsplit,
        nnzsplit
    };

    enum class csrgeam_algorithm
    {
        default_algorithm /*!< Default algorithm */
    };

    enum class csrgemm_algorithm
    {
        default_algorithm /*!< Default algorithm */
    };
} // namespace linalg

#endif // LINEAR_ENUMS_H
