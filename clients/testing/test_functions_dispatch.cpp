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

#include "test_functions.h"

bool Testing::test_dispatch(Arguments arg)
{
    switch(arg.solver)
    {
        case Solver::Jacobi:
            return test_classical(ClassicalSolver::Jacobi, arg);
        case Solver::GaussSeidel:
            return test_classical(ClassicalSolver::GaussSeidel, arg);
        case Solver::SOR:
            return test_classical(ClassicalSolver::SOR, arg);
        case Solver::SymmGaussSeidel:
            return test_classical(ClassicalSolver::SymmGaussSeidel, arg);
        case Solver::SSOR:
            return test_classical(ClassicalSolver::SSOR, arg);
        case Solver::CG:
            return test_krylov(KrylovSolver::CG, arg);
        case Solver::BICGSTAB:
            return test_krylov(KrylovSolver::BICGSTAB, arg);
        case Solver::GMRES:
            return test_krylov(KrylovSolver::GMRES, arg);
        case Solver::SAAMG:
            return test_amg(AMGSolver::SAAMG, arg);
        case Solver::RSAMG:
            return test_amg(AMGSolver::RSAMG, arg);
    }

    return false;
}