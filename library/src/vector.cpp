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

#include "trace.h"
#include "utility.h"

#include "backend/backend_vector.h"
#include "backend/device/device_memory.h"
#include "backend/device/device_vector.h"
#include "backend/host/host_memory.h"
#include "backend/host/host_vector.h"

#include <assert.h>
#include <cstdint>

using namespace linalg;

template <typename T>
vector<T>::vector()
{
    this->hvec    = new host_vector<T>();
    this->dvec    = new device_vector<T>();
    this->vec     = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(size_t size)
{
    this->hvec    = new host_vector<T>(size);
    this->dvec    = new device_vector<T>(size);
    this->vec     = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(size_t size, T val)
{
    this->hvec    = new host_vector<T>(size, val);
    this->dvec    = new device_vector<T>(size, val);
    this->vec     = hvec;
    this->on_host = true;
}

template <typename T>
vector<T>::vector(const std::vector<T>& v)
{
    this->hvec    = new host_vector<T>(v);
    this->dvec    = new device_vector<T>(v);
    this->vec     = hvec;
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

    backend b = determine_backend(*this, x);

    if(b != backend::host && b != backend::device)
    {
        std::cout << "Error: parameters to vector<T>::copy_from must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(this->is_on_host())
    {
        return copy_h2h(this->hvec->get_data(), x.get_vec(), x.get_size());
    }
    else
    {
        return copy_d2d(this->dvec->get_data(), x.get_vec(), x.get_size());
    }
}

template <typename T>
void vector<T>::zeros()
{
    ROUTINE_TRACE("vector<T>::zeros");

    if(this->is_on_host())
    {
        return host_fill(this->hvec->get_data(), this->hvec->get_size(), static_cast<T>(0));
    }
    else
    {
        return device_fill(this->dvec->get_data(), this->dvec->get_size(), static_cast<T>(0));
    }
}

template <typename T>
void vector<T>::ones()
{
    ROUTINE_TRACE("vector<T>::ones");

    if(this->is_on_host())
    {
        return host_fill(this->hvec->get_data(), this->hvec->get_size(), static_cast<T>(1));
    }
    else
    {
        return device_fill(this->dvec->get_data(), this->dvec->get_size(), static_cast<T>(1));
    }
}

template <typename T>
void vector<T>::move_to_device()
{
    ROUTINE_TRACE("vector<T>::move_to_device");

    if(!is_device_available())
    {
        std::cout << "Warning: Device not available. Keeping vector on the host." << std::endl;
        return;
    }

    if(!this->is_on_host())
    {
        std::cout << "Warning: Vector data is already on the device. Doing nothing" << std::endl;
        return;
    }

    size_t old_size = this->vec->get_size();

    this->vec     = this->dvec;
    this->on_host = false;

    if(old_size != this->vec->get_size())
    {
        this->vec->resize(old_size);
    }

    // need to copy data from old vector to current vector
    copy_h2d(this->dvec->get_data(), this->hvec->get_data(), this->hvec->get_size());
}

template <typename T>
void vector<T>::move_to_host()
{
    ROUTINE_TRACE("vector<T>::move_to_host");

    if(this->is_on_host())
    {
        std::cout << "Warning: Vector data is already on the host. Doing nothing" << std::endl;
        return;
    }

    size_t old_size = this->vec->get_size();

    this->vec     = this->hvec;
    this->on_host = true;

    if(old_size != this->vec->get_size())
    {
        this->vec->resize(old_size);
    }

    // need to copy data from old vector to current vector
    copy_d2h(this->hvec->get_data(), this->dvec->get_data(), this->dvec->get_size());
}

template <typename T>
void vector<T>::print_vector(const std::string name) const
{
    ROUTINE_TRACE("vector::print_vector");

    if(!this->is_on_host())
    {
        std::cout << "Matrix must be on the host in order to print to console" << std::endl;
        return;
    }

    std::cout << name << std::endl;
    for(int i = 0; i < this->get_size(); i++)
    {
        std::cout << (*this->vec)[i] << " ";
    }
    std::cout << "" << std::endl;
}

template class linalg::vector<uint32_t>;
template class linalg::vector<int32_t>;
template class linalg::vector<int64_t>;
template class linalg::vector<double>;