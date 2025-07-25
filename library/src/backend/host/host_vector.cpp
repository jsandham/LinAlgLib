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

#include <cstdint>

#include "host_vector.h"

using namespace linalg::host;

template <typename T>
host_vector<T>::host_vector()
{
}
template <typename T>
host_vector<T>::host_vector(size_t size)
{
    hvec.resize(size);
}
template <typename T>
host_vector<T>::host_vector(size_t size, T val)
{
    hvec.resize(size, val);
}
template <typename T>
host_vector<T>::host_vector(const std::vector<T>& vec)
{
    hvec = vec;
}
template <typename T>
host_vector<T>::~host_vector()
{
}
template <typename T>
T* host_vector<T>::get_data()
{
    return hvec.data();
}
template <typename T>
const T* host_vector<T>::get_data() const
{
    return hvec.data();
}
template <typename T>
size_t host_vector<T>::get_size() const
{
    return hvec.size();
}
template <typename T>
void host_vector<T>::clear()
{
    hvec.clear();
}
template <typename T>
void host_vector<T>::resize(size_t size)
{
    hvec.resize(size);
}
template <typename T>
void host_vector<T>::resize(size_t size, T val)
{
    hvec.resize(size, val);
}

template class linalg::host::host_vector<uint32_t>;
template class linalg::host::host_vector<int32_t>;
template class linalg::host::host_vector<int64_t>;
template class linalg::host::host_vector<double>;
