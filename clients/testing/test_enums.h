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

namespace testing
{
    enum class category
    {
        iterative_solvers,
        direct_solvers,
        math,
        primitive,
        csr_matrix,
        Unknown
    };

    enum class fixture
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
        multiply_by_vector,
        multiply_by_matrix,
        triangular_solve,
        compute_incomplete_cholesky_factorization,
        compute_incomplete_cholesky_factorization_dense,
        compute_incomplete_LU_factorization_dense,
        SpTRSV,
        SpGEAM,
        transpose,
        transpose_dense,
        CSRIC0,
        CSRILU0,
        TridiagonalSolver,
        ExclusiveScan,
        Unknown
    };

    enum class backend
    {
        CPU,
        GPU
    };

    enum class uplo
    {
        Lower,
        Upper
    };

    enum class classical_solver
    {
        Jacobi,
        GaussSeidel,
        SOR,
        SymmGaussSeidel,
        SSOR
    };

    enum class krylov_solver
    {
        CG,
        BICGSTAB,
        GMRES
    };

    enum class AMG_solver
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

    enum class pivoting_strategy
    {
        Partial,
        None
    };

    inline std::string category_to_string(category category)
    {
        switch(category)
        {
        case category::iterative_solvers:
            return "iterative_solvers";
        case category::direct_solvers:
            return "direct_solvers";
        case category::math:
            return "math";
        case category::primitive:
            return "primitive";
        case category::csr_matrix:
            return "csr_matrix";
        }

        return "Invalid";
    }

    inline std::string fixture_to_string(fixture fixture)
    {
        switch(fixture)
        {
        case fixture::Jacobi:
            return "Jacobi";
        case fixture::GaussSeidel:
            return "GaussSeidel";
        case fixture::SOR:
            return "SOR";
        case fixture::SymmGaussSeidel:
            return "SymmGaussSeidel";
        case fixture::SSOR:
            return "SSOR";
        case fixture::CG:
            return "CG";
        case fixture::BICGSTAB:
            return "BICGSTAB";
        case fixture::GMRES:
            return "GMRES";
        case fixture::UAAMG:
            return "UAAMG";
        case fixture::SAAMG:
            return "SAAMG";
        case fixture::RSAMG:
            return "RSAMG";
        case fixture::multiply_by_vector:
            return "multiply_by_vector";
        case fixture::multiply_by_matrix:
            return "multiply_by_matrix";
        case fixture::triangular_solve:
            return "triangular_solve";
        case fixture::compute_incomplete_cholesky_factorization:
            return "compute_incomplete_cholesky_factorization";
        case fixture::compute_incomplete_cholesky_factorization_dense:
            return "compute_incomplete_cholesky_factorization_dense";
        case fixture::compute_incomplete_LU_factorization_dense:
            return "compute_incomplete_LU_factorization_dense";
        case fixture::SpGEAM:
            return "SpGEAM";
        case fixture::SpTRSV:
            return "SpTRSV";
        case fixture::transpose:
            return "transpose";
        case fixture::transpose_dense:
            return "transpose_dense";
        case fixture::CSRIC0:
            return "CSRIC0";
        case fixture::CSRILU0:
            return "CSRILU0";
        case fixture::TridiagonalSolver:
            return "TridiagonalSolver";
        case fixture::ExclusiveScan:
            return "ExclusiveScan";
        }

        return "Invalid";
    }

    inline std::string backend_to_string(backend backend)
    {
        switch(backend)
        {
        case backend::CPU:
            return "CPU";
        case backend::GPU:
            return "GPU";
        }

        return "Invalid";
    }

    inline std::string uplo_to_string(uplo uplo)
    {
        switch(uplo)
        {
        case uplo::Lower:
            return "Lower";
        case uplo::Upper:
            return "Upper";
        }

        return "Invalid";
    }

    inline std::string classical_solver_to_string(classical_solver solver)
    {
        switch(solver)
        {
        case classical_solver::Jacobi:
            return "Jacobi";
        case classical_solver::GaussSeidel:
            return "GaussSeidel";
        case classical_solver::SOR:
            return "SOR";
        case classical_solver::SymmGaussSeidel:
            return "SymmGaussSeidel";
        case classical_solver::SSOR:
            return "SSOR";
        }

        return "Invalid";
    }

    inline std::string krylov_solver_to_string(krylov_solver solver)
    {
        switch(solver)
        {
        case krylov_solver::CG:
            return "CG";
        case krylov_solver::BICGSTAB:
            return "BICGSTAB";
        case krylov_solver::GMRES:
            return "GMRES";
        }

        return "Invalid";
    }

    inline std::string amg_solver_to_string(AMG_solver solver)
    {
        switch(solver)
        {
        case AMG_solver::UAAMG:
            return "UAAMG";
        case AMG_solver::SAAMG:
            return "SAAMG";
        case AMG_solver::RSAMG:
            return "RSAMG";
        }

        return "Invalid";
    }

    inline std::string preconditioner_to_string(preconditioner precond)
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

    inline std::string cycle_type_to_string(cycle_type cycle)
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

    inline std::string smoother_type_to_string(smoother_type smoother)
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

    inline std::string pivoting_strategy_to_string(pivoting_strategy strategy)
    {
        switch(strategy)
        {
        case pivoting_strategy::Partial:
            return "Partial";
        case pivoting_strategy::None:
            return "None";
        }

        return "Invalid";
    }
}

#endif
