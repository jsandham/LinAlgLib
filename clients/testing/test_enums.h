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
        unknown
    };

    enum class fixture
    {
        jacobi,
        gauss_seidel,
        SOR,
        symmetric_gauss_seidel,
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
        tridiagonal_solver,
        exclusive_scan,
        ruiz_scaling,
        symmetric_ruiz_scaling,
        unknown
    };

    enum class backend
    {
        CPU,
        GPU
    };

    enum class uplo
    {
        lower,
        upper
    };

    enum class classical_solver
    {
        jacobi,
        gauss_seidel,
        SOR,
        symmetric_gauss_seidel,
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
        jacobi,
        gauss_seidel,
        SOR,
        symmetric_gauss_seidel,
        SSOR,
        ILU,
        IC,
        none
    };

    enum class cycle_type
    {
        vcycle,
        wcycle,
        fcycle,
        none
    };

    enum class smoother_type
    {
        jacobi,
        gauss_seidel,
        symmetric_gauss_seidel,
        SOR,
        SSOR,
        none
    };

    enum class pivoting_strategy
    {
        partial,
        none
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

        return "invalid";
    }

    inline std::string fixture_to_string(fixture fixture)
    {
        switch(fixture)
        {
        case fixture::jacobi:
            return "jacobi";
        case fixture::gauss_seidel:
            return "gauss_seidel";
        case fixture::SOR:
            return "SOR";
        case fixture::symmetric_gauss_seidel:
            return "symmetric_gauss_seidel";
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
        case fixture::tridiagonal_solver:
            return "tridiagonal_solver";
        case fixture::exclusive_scan:
            return "exclusive_scan";
        case fixture::ruiz_scaling:
            return "ruiz_scaling";
        case fixture::symmetric_ruiz_scaling:
            return "symmetric_ruiz_scaling";
        }

        return "invalid";
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

        return "invalid";
    }

    inline std::string uplo_to_string(uplo uplo)
    {
        switch(uplo)
        {
        case uplo::lower:
            return "lower";
        case uplo::upper:
            return "upper";
        }

        return "invalid";
    }

    inline std::string classical_solver_to_string(classical_solver solver)
    {
        switch(solver)
        {
        case classical_solver::jacobi:
            return "jacobi";
        case classical_solver::gauss_seidel:
            return "gauss_seidel";
        case classical_solver::SOR:
            return "SOR";
        case classical_solver::symmetric_gauss_seidel:
            return "symmetric_gauss_seidel";
        case classical_solver::SSOR:
            return "SSOR";
        }

        return "invalid";
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

        return "invalid";
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

        return "invalid";
    }

    inline std::string preconditioner_to_string(preconditioner precond)
    {
        switch(precond)
        {
        case preconditioner::jacobi:
            return "jacobi";
        case preconditioner::gauss_seidel:
            return "gauss_seidel";
        case preconditioner::SOR:
            return "SOR";
        case preconditioner::symmetric_gauss_seidel:
            return "symmetric_gauss_seidel";
        case preconditioner::SSOR:
            return "SSOR";
        case preconditioner::ILU:
            return "ILU";
        case preconditioner::IC:
            return "IC";
        case preconditioner::none:
            return "none";
        }

        return "invalid";
    }

    inline std::string cycle_type_to_string(cycle_type cycle)
    {
        switch(cycle)
        {
        case cycle_type::vcycle:
            return "vcycle";
        case cycle_type::wcycle:
            return "wcycle";
        case cycle_type::fcycle:
            return "fcycle";
        case cycle_type::none:
            return "none";
        }

        return "invalid";
    }

    inline std::string smoother_type_to_string(smoother_type smoother)
    {
        switch(smoother)
        {
        case smoother_type::jacobi:
            return "jacobi";
        case smoother_type::gauss_seidel:
            return "gauss_seidel";
        case smoother_type::symmetric_gauss_seidel:
            return "symmetric_gauss_seidel";
        case smoother_type::SOR:
            return "SOR";
        case smoother_type::SSOR:
            return "SSOR";
        case smoother_type::none:
            return "none";
        }

        return "invalid";
    }

    inline std::string pivoting_strategy_to_string(pivoting_strategy strategy)
    {
        switch(strategy)
        {
        case pivoting_strategy::partial:
            return "partial";
        case pivoting_strategy::none:
            return "none";
        }

        return "invalid";
    }
}

#endif
