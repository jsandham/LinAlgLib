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

#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H

#include <iostream>
#include <vector>

#include "../backend_vector.h"

namespace linalg
{
    namespace device
    {
        template<typename T>
        class device_vector : public backend_vector<T>
        {
            private:
                size_t size;
                T* dvec;

                T temp;

            public:
                device_vector();
                device_vector(size_t size);
                device_vector(size_t size, T val);
                device_vector(const std::vector<T>& vec);
                ~device_vector();

                T& operator[](size_t index) override
                {
                    std::cout << "Error: operator[] not implemented for device vectors." << std::endl;
                    return temp;
                }

                const T& operator[](size_t index) const override
                {
                    std::cout << "Error: operator[] not implemented for device vectors." << std::endl;
                    return temp;
                }

                T* get_data() override;
                const T* get_data() const override;
                size_t get_size() const override;
                void clear() override;
                void resize(size_t size) override;
                void resize(size_t size, T val) override;
        };
    }
}

#endif DEVICE_VECTOR_H