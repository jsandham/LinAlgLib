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

#ifndef VECTOR_H
#define VECTOR_H

#include <string>
#include <vector>

#include "../src/backend/backend_vector.h"

/*! \file
 *  \brief vector.h provides vector class
 */

namespace linalg
{
    template <typename T>
    class backend_vector;

    template <typename T>
    class host_vector;

    template <typename T>
    class device_vector;

    /*! \brief A class for representing and manipulating vectors.
 *
 * \details
 * This class provides functionality for creating, storing, and performing
 * operations on vectors. It supports both host (CPU) and device (GPU) memory
 * management for efficient computation in different environments.
 */
    template <typename T>
    class vector
    {
    private:
        host_vector<T>*    hvec;
        device_vector<T>*  dvec;
        backend_vector<T>* vec;

        /*! \brief Flag indicating if the vector data is currently on the host (CPU) or device (GPU). */
        bool on_host;

    public:
        /*! \brief Default constructor.
     * Initializes an empty vector with a size of 0.
     */
        vector();

        /*! \brief Constructor to create a vector of a specified size.
     * \param size The desired number of elements in the vector.
     */
        vector(size_t size);

        vector(size_t size, T val);

        /*! \brief Constructor to initialize a vector from a `std::vector<double>`.
     * \param vec A `std::vector<double>` whose elements will be copied into this vector.
     */
        vector(const std::vector<T>& vec);

        /*! \brief Destructor.
     * Cleans up any resources allocated by the vector.
     */
        ~vector();

        /*! \brief Deleted copy constructor.
     * Prevents direct copying of `vector` objects to avoid shallow copies and
     * ensure proper memory management. Use `copy_from` for explicit copying.
     */
        vector(const vector&) = delete;

        /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `vector` to another to avoid shallow copies
     * and ensure proper memory management. Use `copy_from` for explicit copying.
     */
        vector& operator=(const vector&) = delete;

        /*! \brief Overload of the array subscript operator for non-constant access.
     * \param index The index of the element to access.
     * \return A reference to the element at the specified index.
     */
        T& operator[](size_t index)
        {
            return *(vec->get_data() + index);
        }

        /*! \brief Overload of the array subscript operator for constant access.
     * \param index The index of the element to access.
     * \return A constant reference to the element at the specified index.
     */
        const T& operator[](size_t index) const
        {
            return vec->get_data()[index];
        }

        /*! \brief Checks if the vector data is currently stored on the host (CPU).
     * \return `true` if the vector data is on the host, `false` otherwise (e.g., on a device).
     */
        bool is_on_host() const;

        /*! \brief Returns the number of elements in the vector.
     * \return The size of the vector.
     */
        size_t get_size() const;

        /*! \brief Returns a non-constant pointer to the beginning of the vector's underlying data.
     * \return A `double*` to the first element of the vector. This allows modification of the vector.
     */
        T* get_vec();

        /*! \brief Returns a constant pointer to the beginning of the vector's underlying data.
     * \return A `const double*` to the first element of the vector. This prevents modification of the vector.
     */
        const T* get_vec() const;

        /*! \brief Resizes the vector to the specified size.
     * \details If the new size is smaller, elements beyond the new size are truncated.
     * If the new size is larger, new elements are default-initialized.
     * \param size The new desired size of the vector.
     */
        void resize(size_t size);

        void resize(size_t size, T val);

        void clear();

        // void assign(size_t size, T val);

        /*! \brief Copies the contents of another `vector` into this object.
     *
     * This performs a deep copy, ensuring that all elements are duplicated.
     * \param x The source `vector` to copy from.
     */
        void copy_from(const vector& x);

        /*! \brief Sets all elements of the vector to zero. */
        void zeros();

        /*! \brief Sets all elements of the vector to one. */
        void ones();

        void fill(T val);

        /*! \brief Moves the vector data from host memory to device memory (e.g., GPU).
     * \details This method handles the necessary memory transfers if a device is available
     * and `on_host` is true. After this call, `is_on_host()` will return `false`.
     */
        void move_to_device();

        /*! \brief Moves the vector data from device memory to host memory (e.g., CPU).
     * \details This method handles the necessary memory transfers if data is on a device
     * and `on_host` is false. After this call, `is_on_host()` will return `true`.
     */
        void move_to_host();

        void print_vector(const std::string name) const;
    };

}
#endif
