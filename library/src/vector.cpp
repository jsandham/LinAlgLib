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

#include "backend/backend_vector.h"
#include "backend/host/host_vector.h"
#include "backend/device/device_vector.h"

#include <assert.h>

using namespace linalg;

template <typename T>
vector<T>::vector()
{
    this->hvec = new host::host_vector<T>();
    this->dvec = new device::device_vector<T>();
    this->vec = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(size_t size)
{
    this->hvec = new host::host_vector<T>(size);
    this->dvec = new device::device_vector<T>(size);
    this->vec = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(size_t size, T val)
{
    this->hvec = new host::host_vector<T>(size, val);
    this->dvec = new device::device_vector<T>(size, val);
    this->vec = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(const std::vector<T>& v)
{
    this->hvec = new host::host_vector<T>(v);
    this->dvec = new device::device_vector<T>(v);
    this->vec = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::~vector()
{
    delete hvec;
    delete dvec;
}

template <typename T>
bool vector<T>::is_on_host() const
{
    return on_host;
}

template <typename T>
size_t vector<T>::get_size() const
{
    return vec->get_size();
}

template <typename T>
T* vector<T>::get_vec()
{
    return vec->get_data();
}

template <typename T>
const T* vector<T>::get_vec() const
{
    return vec->get_data();
}

template <typename T>
void vector<T>::resize(size_t size)
{
    ROUTINE_TRACE("vector<T>::resize");
    vec->resize(size);
}

template <typename T>
void vector<T>::resize(size_t size, T val)
{
    ROUTINE_TRACE("vector<T>::resize");
    vec->resize(size, val);
}

template <typename T>
void vector<T>::clear()
{
    ROUTINE_TRACE("vector<T>::clear");
    vec->clear();
}

// template <typename T>
// void vector<T>::assign(size_t size, T val)
// {
//     vec->assign(size, val);
// }

template <typename T>
void vector<T>::copy_from(const vector& x)
{
    ROUTINE_TRACE("vector<T>::copy_from");
    copy<T>(*this, x);
}

template <typename T>
void vector<T>::zeros()
{
    ROUTINE_TRACE("vector<T>::zeros");
    fill<T>(*this, (T)0);
}
    
template <typename T>
void vector<T>::ones()
{
    ROUTINE_TRACE("vector<T>::ones");
    fill<T>(*this, (T)1);
}

template <typename T>
void vector<T>::move_to_device()
{
    ROUTINE_TRACE("vector<T>::move_to_device");
    this->vec = this->dvec;
    this->on_host = false;
}

template <typename T>
void vector<T>::move_to_host()
{
    ROUTINE_TRACE("vector<T>::move_to_host");
    this->vec = this->hvec;
    this->on_host = true;
}

template class linalg::vector<uint32_t>;
template class linalg::vector<int32_t>;
template class linalg::vector<int64_t>;
template class linalg::vector<double>;