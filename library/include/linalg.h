//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#ifndef LINALG_H__
#define LINALG_H__

/*! \file
*  \brief linalg.h includes other *.h files and provides sparse iterative linear solvers and eigenvalue solvers
*/

// Classic Linear solvers
#include "IterativeSolvers/Classic/jacobi.h"
#include "IterativeSolvers/Classic/gauss_seidel.h"
#include "IterativeSolvers/Classic/sor.h"
#include "IterativeSolvers/Classic/symmetric_gauss_seidel.h"
#include "IterativeSolvers/Classic/ssor.h"
#include "IterativeSolvers/Classic/richardson.h"

// Krylov Linear solvers
#include "IterativeSolvers/Krylov/gmres.h"
#include "IterativeSolvers/Krylov/cg.h"
#include "IterativeSolvers/Krylov/bicgstab.h"

// Algrbraic multi-grid solvers
#include "IterativeSolvers/AMG/amg_aggregation.h"
#include "IterativeSolvers/AMG/amg_strength.h"
#include "IterativeSolvers/AMG/amg.h"
#include "IterativeSolvers/AMG/rsamg.h"
#include "IterativeSolvers/AMG/rsamg_old.h"
#include "IterativeSolvers/AMG/saamg.h"
#include "IterativeSolvers/AMG/uaamg.h"

// Preconditioners
#include "IterativeSolvers/Preconditioner/preconditioner.h"

// Eigenvalues solvers
#include "EigenValueSolvers/power_iteration.h"

#include "IterativeSolvers/slaf.h"
#include "IterativeSolvers/iter_control.h"

#endif
