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

#include <string>
#include <type_traits> // For std::false_type and std::true_type
#include <utility> // For std::declval and std::void_t

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

// --- Type Trait Helper for the `is_on_host()` method ---
// This is a SFINAE-based trait that checks at compile time if a
// given type `T` has a member function named `is_on_host()`.
template <typename T, typename = void>
struct has_is_on_host_method : std::false_type
{
};

template <typename T>
struct has_is_on_host_method<T, std::void_t<decltype(std::declval<T>().is_on_host())>>
    : std::true_type
{
};

template <typename T>
inline bool check_backend_match(bool first_is_on_host, const T& arg)
{
    // If the type `T` has an `is_on_host` method...
    if constexpr(has_is_on_host_method<T>::value)
    {
        return arg.is_on_host() == first_is_on_host;
    }
    else
    {
        // ...otherwise (for primitive types like int, float),
        // we assume it matches and return true.
        return true;
    }
}

inline backend determine_backend()
{
    return backend::host;
}

// Recursive helper that finds the first valid argument and determines the backend.
template <typename... Rest>
inline backend determine_backend_impl(bool first_is_on_host, const Rest&... rest)
{
    // A fold expression using the helper function to check all subsequent arguments.
    bool rest_equal_to_first = (check_backend_match(first_is_on_host, rest) && ...);
    if(rest_equal_to_first)
    {
        return first_is_on_host ? backend::host : backend::device;
    }
    else
    {
        return backend::invalid;
    }
}

template <typename T, typename... Rest>
inline backend determine_backend(const T& first, const Rest&... rest)
{
    // If the first argument has the method, use it as the baseline.
    if constexpr(has_is_on_host_method<T>::value)
    {
        return determine_backend_impl(first.is_on_host(), rest...);
    }
    else
    {
        // If not, skip this argument and recurse on the rest.
        // This is a C++17 `if constexpr` trick.
        if constexpr(sizeof...(Rest) > 0)
        {
            return determine_backend(rest...);
        }
        else
        {
            // Base case: only primitive types were passed.
            return backend::host;
        }
    }
}

// template <typename T, typename... Rest>
// inline backend determine_backend(const T& first, const Rest&... rest)
// {
//     bool first_is_on_host = first.is_on_host();
//     // bool rest_equal_to_first = ((rest.is_on_host() == first_is_on_host) && ...);
//     bool rest_equal_to_first = (check_backend_match(first_is_on_host, rest) && ...);
//     if(rest_equal_to_first)
//     {
//         return first_is_on_host ? backend::host : backend::device;
//     }
//     else
//     {
//         return backend::invalid;
//     }
// }

template <typename HostFunc, typename DeviceFunc, typename... Args>
inline auto
    backend_dispatch(std::string func, HostFunc host_func, DeviceFunc device_func, Args&&... args)
        -> decltype(host_func(std::forward<Args>(args)...))
{
    backend bend = determine_backend(std::forward<Args>(args)...);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters must all be on host or all be on device" << std::endl;
        return decltype(host_func(std::forward<Args>(args)...)){};
    }

    if(bend == backend::host)
    {
        return host_func(std::forward<Args>(args)...);
    }
    else
    {
        return device_func(std::forward<Args>(args)...);
    }
}

#endif