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
vector2::vector2()
{
}

vector2::vector2(size_t size)
{
    hvec.resize(size);
}

vector2::vector2(const std::vector<double>& vec)
{
    hvec = vec;
    on_host = true;
}

vector2::~vector2()
{

}

bool vector2::is_on_host() const
{
    return on_host;
}

int vector2::get_size() const
{
    return hvec.size();
}

double* vector2::get_vec()
{
    return hvec.data();
}

const double* vector2::get_vec() const
{
    return hvec.data();
}

void vector2::copy_from(const vector2& x)
{
    copy(hvec.data(), x.get_vec(), hvec.size());
}

void vector2::zeros()
{
    fill_with_zeros(hvec.data(), hvec.size());
}
    
void vector2::ones()
{
    fill_with_ones(hvec.data(), hvec.size());
}

void vector2::exclusize_scan()
{
    compute_exclusize_scan(hvec.data(), hvec.size());
}

double vector2::dot(const vector2& x) const
{
    assert(this->get_size() == x.get_size());

    return dot_product(this->get_vec(), x.get_vec(), this->get_size());
}

double vector2::norm_euclid2() const
{
    return norm_euclid(hvec.data(), hvec.size());
}
    
double vector2::norm_inf2() const
{
    return norm_inf(hvec.data(), hvec.size());
}

void vector2::resize(size_t size)
{
    hvec.resize(size);
}

void vector2::move_to_device()
{

}

void vector2::move_to_host()
{

}