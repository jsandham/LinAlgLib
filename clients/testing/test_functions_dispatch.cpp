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

static std::string correct_filename(const std::string& filename)
{
#if defined(_WIN32) || defined(WIN32)
    return "../" + filename;
#else
    return filename;
#endif
}

namespace Testing
{
    static bool solver_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
            case Fixture::Jacobi:
                return test_classical(ClassicalSolver::Jacobi, arg);
            case Fixture::GaussSeidel:
                return test_classical(ClassicalSolver::GaussSeidel, arg);
            case Fixture::SOR:
                return test_classical(ClassicalSolver::SOR, arg);
            case Fixture::SymmGaussSeidel:
                return test_classical(ClassicalSolver::SymmGaussSeidel, arg);
            case Fixture::SSOR:
                return test_classical(ClassicalSolver::SSOR, arg);
            case Fixture::CG:
                return test_krylov(KrylovSolver::CG, arg);
            case Fixture::BICGSTAB:
                return test_krylov(KrylovSolver::BICGSTAB, arg);
            case Fixture::GMRES:
                return test_krylov(KrylovSolver::GMRES, arg);
            case Fixture::UAAMG:
                return test_amg(AMGSolver::UAAMG, arg);
            case Fixture::SAAMG:
                return test_amg(AMGSolver::SAAMG, arg);
            case Fixture::RSAMG:
                return test_amg(AMGSolver::RSAMG, arg);
        }

        return false;
    }

    static bool math_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
            case Fixture::SpMV:
                return test_spmv(arg);
            case Fixture::SpGEMM:
                return test_spgemm(arg);
            case Fixture::SpGEAM:
                return test_spgeam(arg);
        }

        return false;
    }

    static bool primitive_test_dispatch(Arguments arg)
    {
        switch(arg.fixture)
        {
            case Fixture::ExclusiveScan:
                return test_exclusive_scan(arg);
        }
        
        return false;
    }
}

bool Testing::test_dispatch(Arguments arg)
{
    arg.filename = correct_filename(arg.filename);

    switch(arg.category)
    {
        case Category::IterativeSolvers:
           return solver_test_dispatch(arg);
        case Category::Math:
           return math_test_dispatch(arg);
        case Category::Primitive:
           return primitive_test_dispatch(arg);
    }

    return false;
}