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

#ifndef TEST_KRYLOV_H__
#define TEST_KRYLOV_H__

#include <string>

namespace Testing
{
    enum class KrylovSolver
    {
        CG,
        BICGSTAB
    };

    enum class Preconditioner
    {
        Jacobi,
        ILU,
        IC,
        None
    };

    inline std::string KrylovSolverToString(KrylovSolver solver)
    {
        switch(solver)
        {
            case KrylovSolver::CG:
                return "CG";
            case KrylovSolver::BICGSTAB:
                return "BICGSTAB";
        }

        return "Invalid";
    }

    inline std::string PreconditionerToString(Preconditioner precond)
    {
        switch(precond)
        {
            case Preconditioner::Jacobi:
                return "Jacobi";
            case Preconditioner::ILU:
                return "ILU";
            case Preconditioner::IC:
                return "IC";
            case Preconditioner::None:
                return "None";
        }

        return "Invalid";
    }

    bool test_krylov(Testing::KrylovSolver solver, Testing::Preconditioner precond, const std::string &matrix_file);
}

#endif