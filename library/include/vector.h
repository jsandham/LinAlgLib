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

#ifndef VECTOR_H
#define VECTOR_H

#include <vector>

/*! \file
 *  \brief vector.h provides vector class
 */

class vector2
{
private:
    std::vector<double> hvec;

    bool on_host;

public:
    vector2();
    vector2(size_t size);
    vector2(const std::vector<double>& vec);
    ~vector2();

    vector2 (const vector2&) = delete;
    vector2& operator= (const vector2&) = delete;

    double& operator[](size_t index) 
    {
        return hvec[index];
    }

    const double& operator[](size_t index) const 
    {
        return hvec[index];
    }

    bool is_on_host() const;

    int get_size() const;

    double* get_vec();
    const double* get_vec() const;

    void copy_from(const vector2& x);
    void zeros();
    void ones();
    void exclusize_scan();

    double dot(const vector2& x) const;
    double norm_euclid2() const;
    double norm_inf2() const;

    void resize(size_t size);
    void move_to_device();
    void move_to_host();
};


#endif
