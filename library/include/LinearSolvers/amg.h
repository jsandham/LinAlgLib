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
// The above copyright noticeand this permission notice shall be included in all
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


#ifndef AMG_H
#define AMG_H

#include <vector>

/*! \file
*  \brief amg.h provides interface for algebraic multigrid solver
*/
struct csr_matrix
{
	int m;
	int n;
	int nnz;
	std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	std::vector<double> csr_val;
};

struct heirarchy
{
	// Prolongation matrices 
	std::vector<csr_matrix> prolongations;

	// Restriction matrices
	std::vector<csr_matrix> restrictions;

	// Coarse level Amatrices
	std::vector<csr_matrix> A_cs;

	int total_levels;
};

enum class Cycle
{
	Vcycle,
	Wcycle,
	Fcycle
};

enum class Smoother
{
	Jacobi,
	Gauss_Siedel,
	Symm_Gauss_Siedel,
	SOR,
	SSOR
};

void amg_solve(const heirarchy& hierarchy, double* x, const double* b, int n1, int n2, double tol, Cycle cycle, Smoother smoother);

#endif