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

#include "../include/vector.h"
#include "../include/linalg_math.h"

#include "trace.h"

#include <assert.h>

using namespace linalg;

template <typename T>
vector<T>::vector()
{
    this->on_host = true;
}

template <typename T>
vector<T>::vector(size_t size)
{
    this->on_host = true;
    hvec.resize(size);
}

template <typename T>
vector<T>::vector(size_t size, T val)
{
    this->on_host = true;
    hvec.resize(size, val);
}

template <typename T>
vector<T>::vector(const std::vector<T>& vec)
{
    hvec = vec;
    on_host = true;
}

template <typename T>
vector<T>::~vector()
{

}

template <typename T>
bool vector<T>::is_on_host() const
{
    return on_host;
}

template <typename T>
size_t vector<T>::get_size() const
{
    return hvec.size();
}

template <typename T>
T* vector<T>::get_vec()
{
    return hvec.data();
}

template <typename T>
const T* vector<T>::get_vec() const
{
    return hvec.data();
}

template <typename T>
void vector<T>::copy_from(const vector& x)
{
    ROUTINE_TRACE("vector<T>::copy_from");
    copy(*this, x);
}

template <typename T>
void vector<T>::zeros()
{
    ROUTINE_TRACE("vector<T>::zeros");
    fill_with_zeros(*this);
}
    
template <typename T>
void vector<T>::ones()
{
    ROUTINE_TRACE("vector<T>::ones");
    fill_with_ones(*this);
}

template <typename T>
void vector<T>::resize(size_t size)
{
    hvec.resize(size);
}

template <typename T>
void vector<T>::resize(size_t size, T val)
{
    hvec.resize(size, val);
}

template <typename T>
void vector<T>::clear()
{
    hvec.clear();
}

template <typename T>
void vector<T>::assign(size_t size, T val)
{
    hvec.assign(size, val);
}

template <typename T>
void vector<T>::move_to_device()
{
    ROUTINE_TRACE("vector<T>::move_to_device");
}

template <typename T>
void vector<T>::move_to_host()
{
    ROUTINE_TRACE("vector<T>::move_to_host");
}

template class vector<uint32_t>;
template class vector<int32_t>;
template class vector<int64_t>;
template class vector<double>;