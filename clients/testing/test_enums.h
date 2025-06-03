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

#ifndef TEST_ENUMS_H__
#define TEST_ENUMS_H__

#include <string>
#include <linalg.h>

namespace Testing
{
    enum class Solver
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        SSOR,
        CG,
        BICGSTAB,
        GMRES,
        UAAMG,
        SAAMG,
        RSAMG
    };

    enum class ClassicalSolver
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        SSOR
    };

    enum class KrylovSolver
    {
        CG,
        BICGSTAB,
        GMRES
    };

    enum class AMGSolver
    {
        UAAMG,
        SAAMG,
        RSAMG
    };

    enum class Preconditioner
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        ILU,
        IC,
        None
    };
    
    inline std::string SolverToString(Solver solver)
    {
        switch(solver)
        {
            case Solver::Jacobi:
                return "Jacobi";
            case Solver::GaussSeidel:
                return "GaussSeidel";
            case Solver::SOR:
                return "SOR";
            case Solver::SymmGaussSeidel:
                return "SymmGaussSeidel";
            case Solver::SSOR:
                return "SSOR";
            case Solver::CG:
                return "CG";
            case Solver::BICGSTAB:
                return "BICGSTAB";
            case Solver::GMRES:
                return "GMRES";
            case Solver::UAAMG:
                return "UAAMG";
            case Solver::SAAMG:
                return "SAAMG";
            case Solver::RSAMG:
                return "RSAMG";
        }

        return "Invalid";
    }

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

        return "Invalid";
    }

    inline std::string KrylovSolverToString(KrylovSolver solver)
    {
        switch(solver)
        {
            case KrylovSolver::CG:
                return "CG";
            case KrylovSolver::BICGSTAB:
                return "BICGSTAB";
            case KrylovSolver::GMRES:
                return "GMRES";
        }

        return "Invalid";
    }

    inline std::string PreconditionerToString(Preconditioner precond)
    {
        switch(precond)
        {
            case Preconditioner::Jacobi:
                return "Jacobi";
            case Preconditioner::GaussSeidel:
                return "GaussSeidel";
            case Preconditioner::SOR:
                return "SOR";
            case Preconditioner::SymmGaussSeidel:
                return "SymmGaussSeidel";
            case Preconditioner::ILU:
                return "ILU";
            case Preconditioner::IC:
                return "IC";
            case Preconditioner::None:
                return "None";
        }

        return "Invalid";
    }

    inline std::string AMGSolverToString(AMGSolver solver)
    {
        switch(solver)
        {
            case AMGSolver::UAAMG:
                return "UAAMG";
            case AMGSolver::SAAMG:
                return "SAAMG";
            case AMGSolver::RSAMG:
                return "RSAMG";
        }

        return "Invalid";
    }

    inline std::string CycleToString(linalg::Cycle cycle)
    {
        switch(cycle)
        {
            case linalg::Cycle::Vcycle:
                return "Vcycle";
            case linalg::Cycle::Wcycle:
                return "Wcycle";
            case linalg::Cycle::Fcycle:
                return "Fcycle";
        }

        return "Invalid";
    }

    inline std::string SmootherToString(linalg::Smoother smoother)
    {
        switch(smoother)
        {
            case linalg::Smoother::Jacobi:
                return "Jacobi";
            case linalg::Smoother::Gauss_Seidel:
                return "Gauss_Seidel";
            case linalg::Smoother::Symm_Gauss_Seidel:
                return "Symm_Gauss_Seidel";
            case linalg::Smoother::SOR:
                return "SOR";
            case linalg::Smoother::SSOR:
                return "SSOR";
        }

        return "Invalid";
    }
}

#endif