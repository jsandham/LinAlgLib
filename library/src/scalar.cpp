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

#include "../include/scalar.h"
#include "../include/linalg_math.h"

#include "trace.h"

#include "backend/host/host_scalar.h"
#include "backend/device/device_scalar.h"

#include <assert.h>

using namespace linalg;

template <typename T>
const scalar<T>& scalar<T>::one() 
{
    static scalar<T> instance(static_cast<T>(1));
    return instance;
}

template <typename T>
const scalar<T>& scalar<T>::zero() 
{
    static scalar<T> instance(static_cast<T>(0));
    return instance;
}

template <typename T>
scalar<T>::scalar()
{
    this->on_host = true;
    hval = 0;
}

template <typename T>
scalar<T>::scalar(T val)
{
    this->on_host = true;
    hval = val;
}

template <typename T>
scalar<T>::~scalar()
{

}

// Copy assignment
template <typename T>
scalar<T>& scalar<T>::operator=(const scalar<T>& other) 
{
    if (this != &other) 
    {
        hval = other.hval;
        on_host = other.on_host;
    }
    return *this;
}

template <typename T>
scalar<T>& scalar<T>::operator=(const T& other) 
{
    hval = other;
    on_host = true;
    return *this;
}

template <typename T>
scalar<T>& scalar<T>::operator*=(const T& rhs) 
{
    this->hval *= rhs;
    return *this;
}

template <typename T>
scalar<T>& scalar<T>::operator/=(const T& rhs) 
{
    this->hval /= rhs;
    return *this;
}

template <typename T>
bool scalar<T>::is_on_host() const
{
    return on_host;
}

template <typename T>
T* scalar<T>::get_val()
{
    return &hval;
}

template <typename T>
const T* scalar<T>::get_val() const
{
    return &hval;
}

template <typename T>
void scalar<T>::copy_from(const scalar& x)
{
    ROUTINE_TRACE("scalar<T>::copy_from");
    
}

template <typename T>
void scalar<T>::move_to_device()
{
    ROUTINE_TRACE("scalar<T>::move_to_device");
}

template <typename T>
void scalar<T>::move_to_host()
{
    ROUTINE_TRACE("scalar<T>::move_to_host");
}

template class scalar<double>;