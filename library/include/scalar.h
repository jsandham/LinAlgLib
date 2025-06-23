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

#ifndef SCALAR_H
#define SCALAR_H

#include <iostream>

/*! \file
 *  \brief scalar.h provides scalar class
 */

namespace linalg
{
    namespace host
    {
        template<typename T>
        class host_scalar;
    }

    namespace device
    {
        template <typename T>
        class device_scalar;
    }

template<typename T>
class scalar
{
private:
    //host::host_scalar<T> hscalar;
    //device::device_scalar<T> dscalar;

    T hval;

    bool on_host;

public:
    scalar();
    scalar(T val);
    ~scalar();

    // scalar(const scalar& other);
    // scalar (const scalar&) = delete;

    scalar& operator=(const scalar& other);
    scalar& operator=(const T& other); 

    //scalar operator/(scalar lhs, const scalar& rhs);
    //scalar operator*(scalar lhs, const scalar& rhs);

    scalar& operator*=(const T& rhs);
    scalar& operator/=(const T& rhs);


    bool is_on_host() const;

    T* get_val();
    const T* get_val() const;

    void copy_from(const scalar& x);
    void move_to_device();
    void move_to_host();

    static const scalar<T>& one();
    static const scalar<T>& zero();

    friend std::ostream& operator<<(std::ostream& os, const scalar& s) {
        if (s.is_on_host()) {
            os << *(s.get_val());
        } 
        else 
        {
            os << "[Device value not accessible]";
        }
        return os;
    }
};


template <typename T>
scalar<T> operator*(scalar<T> lhs, const T& rhs) 
{
    lhs *= rhs;
    return lhs;
}

template <typename T>
scalar<T> operator*(const T& lhs, scalar<T> rhs) 
{
    std::cout << "operator* lhs: " << lhs << " rhs: " << rhs << std::endl;
    rhs *= lhs;
    return rhs;
}

template <typename T>
scalar<T> operator+(scalar<T> lhs, const T& rhs) 
{
    lhs += rhs;
    return lhs;
}

template <typename T>
scalar<T> operator+(const T& lhs, scalar<T> rhs) 
{
    std::cout << "operator* lhs: " << lhs << " rhs: " << rhs << std::endl;
    rhs += lhs;
    return rhs;
}

}

#endif