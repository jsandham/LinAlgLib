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

#ifndef TEST_FUNCTIONS_H__
#define TEST_FUNCTIONS_H__

#include "test_arguments.h"

namespace Testing
{
    bool test_dispatch(Arguments arg);

    // Iterative solvers
    bool test_classical(ClassicalSolver solver_type, Arguments arg);
    bool test_krylov(KrylovSolver solver_type, Arguments arg);
    bool test_amg(AMGSolver solver_type, Arguments arg);

    // Math testing
    bool test_spmv(Arguments arg);
    bool test_spgemm(Arguments arg);
    bool test_spgeam(Arguments arg);

    // Primitive
    bool test_exclusive_scan(Arguments arg);

} // namespace Testing

#endif