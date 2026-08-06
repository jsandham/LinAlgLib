//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#include "../include/linalg_buffers.h"

#include "backend/device/device_memory.h"

using namespace linalg;

template <typename T>
dp_opt_buffer<T>::dp_opt_buffer()
    : data(nullptr)
    , on_host(true)
{
}

template <typename T>
dp_opt_buffer<T>::~dp_opt_buffer()
{
    free_buffer();
}

template <typename T>
void dp_opt_buffer<T>::allocate_buffer(size_t size)
{
    device_allocate<T>(&data, 256);
}

template <typename T>
void dp_opt_buffer<T>::free_buffer()
{
    if(data)
    {
        device_free(data);
        data = nullptr;
    }
}

template <typename T>
T* dp_opt_buffer<T>::get_buffer()
{
    return data;
}

template class dp_opt_buffer<float>;
template class dp_opt_buffer<double>;
