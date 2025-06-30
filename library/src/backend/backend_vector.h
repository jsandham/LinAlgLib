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

#ifndef BACKEND_VECTOR_H
#define BACKEND_VECTOR_H

namespace linalg
{
    template <typename T>
    class backend_vector
    {
        public:
            backend_vector() = default;
            backend_vector(const backend_vector&) = delete;
            backend_vector& operator=(const backend_vector&) = delete;

            virtual T& operator[](size_t index) = 0;
            virtual const T& operator[](size_t index) const = 0;

            virtual ~backend_vector(){};
            virtual T* get_data() = 0;
            virtual const T* get_data() const = 0;
            virtual size_t get_size() const = 0;
            virtual void clear() = 0;
            virtual void resize(size_t size) = 0;
            virtual void resize(size_t size, T val) = 0;

    };
}
#endif