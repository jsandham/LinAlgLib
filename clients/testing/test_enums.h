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

#include <linalg.h>
#include <string>

namespace Testing
{
    enum class Category
    {
        IterativeSolvers,
        Math,
        Primitive,
        Unknown
    };

    enum class Fixture
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
        RSAMG,
        SpMV,
        SpTRSV,
        SpGEMM,
        SpGEAM,
        Transpose,
        CSRIC0,
        CSRILU0,
        TridiagonalSolver,
        ExclusiveScan,
        Unknown
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

    enum class preconditioner
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        SSOR,
        ILU,
        IC,
        None
    };

    enum class cycle_type
    {
        Vcycle,
        Wcycle,
        Fcycle,
        None
    };

    enum class smoother_type
    {
        Jacobi,
        Gauss_Seidel,
        Symm_Gauss_Seidel,
        SOR,
        SSOR,
        None
    };

    inline std::string CategoryToString(Category category)
    {
        switch(category)
        {
        case Category::IterativeSolvers:
            return "IterativeSolvers";
        case Category::Math:
            return "Math";
        case Category::Primitive:
            return "Primitive";
        }

        return "Invalid";
    }

    inline std::string FixtureToString(Fixture fixture)
    {
        switch(fixture)
        {
        case Fixture::Jacobi:
            return "Jacobi";
        case Fixture::GaussSeidel:
            return "GaussSeidel";
        case Fixture::SOR:
            return "SOR";
        case Fixture::SymmGaussSeidel:
            return "SymmGaussSeidel";
        case Fixture::SSOR:
            return "SSOR";
        case Fixture::CG:
            return "CG";
        case Fixture::BICGSTAB:
            return "BICGSTAB";
        case Fixture::GMRES:
            return "GMRES";
        case Fixture::UAAMG:
            return "UAAMG";
        case Fixture::SAAMG:
            return "SAAMG";
        case Fixture::RSAMG:
            return "RSAMG";
        case Fixture::SpMV:
            return "SpMV";
        case Fixture::SpGEMM:
            return "SpGEMM";
        case Fixture::SpGEAM:
            return "SpGEAM";
        case Fixture::SpTRSV:
            return "SpTRSV";
        case Fixture::Transpose:
            return "Transpose";
        case Fixture::CSRIC0:
            return "CSRIC0";
        case Fixture::CSRILU0:
            return "CSRILU0";
        case Fixture::TridiagonalSolver:
            return "TridiagonalSolver";
        case Fixture::ExclusiveScan:
            return "ExclusiveScan";
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

    inline std::string PreconditionerToString(preconditioner precond)
    {
        switch(precond)
        {
        case preconditioner::Jacobi:
            return "Jacobi";
        case preconditioner::GaussSeidel:
            return "GaussSeidel";
        case preconditioner::SOR:
            return "SOR";
        case preconditioner::SymmGaussSeidel:
            return "SymmGaussSeidel";
        case preconditioner::SSOR:
            return "SSOR";
        case preconditioner::ILU:
            return "ILU";
        case preconditioner::IC:
            return "IC";
        case preconditioner::None:
            return "None";
        }

        return "Invalid";
    }

    inline std::string CycleTypeToString(cycle_type cycle)
    {
        switch(cycle)
        {
        case cycle_type::Vcycle:
            return "Vcycle";
        case cycle_type::Wcycle:
            return "Wcycle";
        case cycle_type::Fcycle:
            return "Fcycle";
        case cycle_type::None:
            return "None";
        }

        return "Invalid";
    }

    inline std::string SmootherTypeToString(smoother_type smoother)
    {
        switch(smoother)
        {
        case smoother_type::Jacobi:
            return "Jacobi";
        case smoother_type::Gauss_Seidel:
            return "Gauss_Seidel";
        case smoother_type::Symm_Gauss_Seidel:
            return "Symm_Gauss_Seidel";
        case smoother_type::SOR:
            return "SOR";
        case smoother_type::SSOR:
            return "SSOR";
        case smoother_type::None:
            return "None";
        }

        return "Invalid";
    }
}

#endif
