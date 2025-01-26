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

#ifndef TEST_AMG_H__
#define TEST_AMG_H__

#include <string>
#include "linalg.h"

namespace Testing
{
    enum class AMGSolver
    {
        SAAMG,
        RSAMG
    };

    inline std::string AMGSolverToString(AMGSolver solver)
    {
        switch(solver)
        {
            case AMGSolver::SAAMG:
                return "SAAMG";
            case AMGSolver::RSAMG:
                return "RSAMG";
        }

        return "Invalid";
    }

    inline std::string CycleToString(Cycle cycle)
    {
        switch(cycle)
        {
            case Cycle::Vcycle:
                return "Vcycle";
            case Cycle::Wcycle:
                return "Wcycle";
            case Cycle::Fcycle:
                return "Fcycle";
        }

        return "Invalid";
    }

    inline std::string SmootherToString(Smoother smoother)
    {
        switch(smoother)
        {
            case Smoother::Jacobi:
                return "Jacobi";
            case Smoother::Gauss_Siedel:
                return "Gauss_Siedel";
            case Smoother::Symm_Gauss_Siedel:
                return "Symm_Gauss_Siedel";
            case Smoother::SOR:
                return "SOR";
            case Smoother::SSOR:
                return "SSOR";
        }

        return "Invalid";
    }

    bool test_amg(Testing::AMGSolver solver, int presmoothing, int postsmoothing, Cycle cycle, Smoother smoother, const std::string &matrix_file);
}

#endif