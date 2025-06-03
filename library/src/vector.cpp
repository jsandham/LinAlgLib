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
vector::vector()
{
}

vector::vector(size_t size)
{
    hvec.resize(size);
}

vector::vector(const std::vector<double>& vec)
{
    hvec = vec;
    on_host = true;
}

vector::~vector()
{

}

bool vector::is_on_host() const
{
    return on_host;
}

int vector::get_size() const
{
    return hvec.size();
}

double* vector::get_vec()
{
    return hvec.data();
}

const double* vector::get_vec() const
{
    return hvec.data();
}

void vector::copy_from(const vector& x)
{
    copy(hvec.data(), x.get_vec(), hvec.size());
}

void vector::zeros()
{
    fill_with_zeros(hvec.data(), hvec.size());
}
    
void vector::ones()
{
    fill_with_ones(hvec.data(), hvec.size());
}

void vector::exclusize_scan()
{
    compute_exclusize_scan(hvec.data(), hvec.size());
}

double vector::dot(const vector& x) const
{
    assert(this->get_size() == x.get_size());

    return dot_product(this->get_vec(), x.get_vec(), this->get_size());
}

double vector::norm_euclid2() const
{
    return norm_euclid(hvec.data(), hvec.size());
}
    
double vector::norm_inf2() const
{
    return norm_inf(hvec.data(), hvec.size());
}

void vector::resize(size_t size)
{
    hvec.resize(size);
}

void vector::move_to_device()
{

}

void vector::move_to_host()
{

}