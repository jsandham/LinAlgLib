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
#include "../include/slaf.h"

#include <assert.h>

using namespace linalg;

template <typename T>
vector<T>::vector()
{
}

template <typename T>
vector<T>::vector(size_t size)
{
    hvec.resize(size);
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
    copy(hvec.data(), x.get_vec(), hvec.size());
}

template <typename T>
void vector<T>::zeros()
{
    fill_with_zeros(hvec.data(), hvec.size());
}
    
template <typename T>
void vector<T>::ones()
{
    fill_with_ones(hvec.data(), hvec.size());
}

template <typename T>
void vector<T>::exclusize_scan()
{
    compute_exclusize_scan(hvec.data(), hvec.size());
}

template <typename T>
T vector<T>::dot(const vector& x) const
{
    assert(this->get_size() == x.get_size());

    return dot_product(this->get_vec(), x.get_vec(), this->get_size());
}

template <typename T>
T vector<T>::norm_euclid2() const
{
    return norm_euclid(hvec.data(), hvec.size());
}
    
template <typename T>
T vector<T>::norm_inf2() const
{
    return norm_inf(hvec.data(), hvec.size());
}

template <typename T>
void vector<T>::resize(size_t size)
{
    hvec.resize(size);
}

template <typename T>
void vector<T>::move_to_device()
{

}

template <typename T>
void vector<T>::move_to_host()
{

}

// template class vector<int>;
template class vector<double>;