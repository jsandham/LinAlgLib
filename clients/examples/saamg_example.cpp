//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include <iostream>
#include <vector>

#include "linalg.h"
#include "utility.h"

int main()
{
    csr_matrix A;
    A.read_mtx("../matrices/SPD/ex5/ex5.mtx");

    // Solution vector
    vector x(A.get_m());
    x.zeros();

    // Righthand side vector
    vector b(A.get_m());
    b.ones();

    hierarchy hierachy;
    saamg_setup(A, 10, hierachy);

    iter_control control;

    int cycles = amg_solve(hierachy, x, b, 10, 10, Cycle::Vcycle, Smoother::Gauss_Seidel, control);
    // int cycles = amg_solve(hierachy, x.data(), b.data(), 2, 2, Cycle::Wcycle, Smoother::Gauss_Seidel, control);
    // int cycles = amg_solve(hierachy, x.data(), b.data(), 2, 2, Cycle::Wcycle, Smoother::SOR, control);

    // Print solution
    std::cout << "x" << std::endl;
    for (int i = 0; i < x.get_size(); i++)
    {
        std::cout << x[i] << " ";
    }
    std::cout << "" << std::endl;

    return 0;
}