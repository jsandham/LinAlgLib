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

#include "device_vector.h"
#include "device_memory.h"

using namespace linalg;

template <typename T>
device_vector<T>::device_vector()
{
    this->dvec = nullptr;
    this->size = 0;
}
template <typename T>
device_vector<T>::device_vector(size_t size)
{
    this->size = size;
    device_allocate(&dvec, size);
}

template <typename T>
device_vector<T>::device_vector(size_t size, T val)
{
    this->size = size;
    device_allocate(&dvec, size);
    device_fill(dvec, size, val);
}

template <typename T>
device_vector<T>::device_vector(const std::vector<T>& vec)
{
    size = vec.size();
    device_allocate(&dvec, size);
    copy_h2d(dvec, vec.data(), size);
}

template <typename T>
device_vector<T>::~device_vector()
{
    size = 0;
    device_free(dvec);
}
template <typename T>
T* device_vector<T>::get_data()
{
    return dvec;
}
template <typename T>
const T* device_vector<T>::get_data() const
{
    return dvec;
}
template <typename T>
size_t device_vector<T>::get_size() const
{
    return size;
}
template <typename T>
void device_vector<T>::clear()
{
    // hvec.clear();
}
template <typename T>
void device_vector<T>::resize(size_t size)
{
    if(this->size != size)
    {
        device_free(dvec);
        this->size = size;
        device_allocate(&dvec, size);
    }
}
template <typename T>
void device_vector<T>::resize(size_t size, T val)
{
    if(this->size != size)
    {
        device_free(dvec);
        this->size = size;
        device_allocate(&dvec, size);
        device_fill(dvec, size, val);
    }
}

template class linalg::device_vector<uint32_t>;
template class linalg::device_vector<int32_t>;
template class linalg::device_vector<int64_t>;
template class linalg::device_vector<double>;
