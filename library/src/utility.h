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

#ifndef UTILITY_H
#define UTILITY_H

inline constexpr bool is_device_available()
{
#ifdef LINALGLIB_HAS_CUDA
    return true;
#else
    return false;
#endif
}

enum class backend
{
    host,
    device,
    invalid
};

template <typename T, typename... Rest>
inline backend determine_backend(const T& first, const Rest&... rest)
{
    bool first_is_on_host    = first.is_on_host();
    bool rest_equal_to_first = ((rest.is_on_host() == first_is_on_host) && ...);
    if(rest_equal_to_first)
    {
        return first_is_on_host ? backend::host : backend::device;
    }
    else
    {
        return backend::invalid;
    }
}

#endif