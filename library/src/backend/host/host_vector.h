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

#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include <vector>

#include "../backend_vector.h"

namespace linalg
{
    namespace host
    {
        template <typename T>
        class host_vector : public backend_vector<T>
        {
        private:
            std::vector<T> hvec;

        public:
            host_vector();
            host_vector(size_t size);
            host_vector(size_t size, T val);
            host_vector(const std::vector<T>& vec);
            ~host_vector();

            T& operator[](size_t index) override
            {
                return hvec[index];
            }

            const T& operator[](size_t index) const override
            {
                return hvec[index];
            }

            T*       get_data() override;
            const T* get_data() const override;
            size_t   get_size() const override;
            void     clear() override;
            void     resize(size_t size) override;
            void     resize(size_t size, T val) override;
        };
    }

}

#endif