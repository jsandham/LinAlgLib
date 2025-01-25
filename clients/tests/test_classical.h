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

#ifndef TEST_CLASSICAL_H__
#define TEST_CLASSICAL_H__

#include <string>

namespace Testing
{
    enum class ClassicalSolver
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        SSOR
    };

    inline std::string ClassicalSolverToString(ClassicalSolver solver)
    {
        switch(solver)
        {
            case ClassicalSolver::Jacobi:
                return "Jacobi";
            case ClassicalSolver::GaussSeidel:
                return "GaussSeidel";
            case ClassicalSolver::SOR:
                return "SOR";
            case ClassicalSolver::SymmGaussSeidel:
                return "SymmGaussSeidel";
            case ClassicalSolver::SSOR:
                return "SSOR";
        }
    }

    bool test_classical(ClassicalSolver solver, std::string &matrix_file);
} // namespace Testing

#endif