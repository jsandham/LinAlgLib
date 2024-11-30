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

#include"iostream"
#include"../../include/LinearSolvers/jgs.h"
#include"../../include/LinearSolvers/slaf.h"
#include"math.h"

//********************************************************************************
//
// Jacobian, Guass Seidel, SOR, Symmetric Gauss Seidel, and Symmetric SOR 
//
//********************************************************************************

#define DEBUG 1


//-------------------------------------------------------------------------------
// jacobi method
//-------------------------------------------------------------------------------
void jacobi_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* xold, const double* b, const int n)
{
	double sigma;
	double ajj;
	for (int j = 0; j < n; j++)
	{
		sigma = 0.0;
		ajj = 0.0;   //diagonal entry a_jj
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * xold[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = (b[j] - sigma) / ajj;
	}
}

int jac(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double tol, const int max_iter)
{
	//copy of x
	double *xold = new double[n];
	for(int i = 0; i < n; i++)
	{
		xold[i] = x[i];
	}

	int ii = 0;
	double err = 1.0;
	while(err > tol && ii < max_iter)
	{
		//Jacobi iteration
		jacobi_iteration(csr_row_ptr, csr_col_ind, csr_val, x, xold, b, n);
		
		for(int i = 0; i < n; i++)
		{
			xold[i] = x[i];
		}

		err = error(csr_row_ptr, csr_col_ind, csr_val, xold, b, n);
		#if(DEBUG)
			std::cout << "error: " << err << std::endl;
		#endif

		ii++;
	}

	delete[] xold;

	return err > tol ? -1 : ii;
}



//-------------------------------------------------------------------------------
// gauss-seidel method
//-------------------------------------------------------------------------------
void gauss_siedel_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, const int n)
{
	for (int j = 0; j < n; j++)
	{
		double sigma = 0.0;
		double ajj = 0.0;   //diagonal entry a_jj

		int row_start = csr_row_ptr[j];
		int row_end = csr_row_ptr[j + 1];

		for (int k = row_start; k < row_end; k++)
		{
			int col = csr_col_ind[k];
			double val = csr_val[k];
			if (col != j)
			{
				sigma = sigma + val * x[col];
			}
			else
			{
				ajj = val;
			}
		}
		x[j] = (b[j] - sigma) / ajj;
	}
}

int gs(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
       const int n, const double tol, const int max_iter)
{
	int ii = 0;
	double err = 1.0;
	while(err > tol && ii < max_iter)
	{
		//Gauss-Seidel iteration
		gauss_siedel_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n);

		//err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
		err = fast_error(csr_row_ptr, csr_col_ind, csr_val, x, b, n, tol);

		#if(DEBUG)
			std::cout<<"error: "<<err<<std::endl;
		#endif

		ii++;
	}

	return err > tol ? -1 : ii;
}

//-------------------------------------------------------------------------------
// successive over-relaxation method
//-------------------------------------------------------------------------------
void sor_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, const int n, const double omega)
{
	double sigma;
	double ajj;
	for (int j = 0; j < n; j++)
	{
		sigma = 0.0;
		ajj = 0.0;   //diagonal entry a_jj
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * x[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = x[j] + omega * ((b[j] - sigma) / ajj - x[j]);
	}
}

int sor(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double omega, const double tol, const int max_iter)
{
	int ii = 0;
	double err = 1.0;
	while(err > tol && ii < max_iter)
	{
		//SOR iteration
		sor_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n, omega);

		err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
		
		#if(DEBUG)
			std::cout<<"error: "<<err<<std::endl;
		#endif

		ii++;
	}

	return err > tol ? -1 : ii;
}


//-------------------------------------------------------------------------------
// symmetric Gauss Seidel method
//-------------------------------------------------------------------------------
void symm_gauss_siedel_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, const int n)
{
	double sigma;
	double ajj;

	//forward pass
	for (int j = 0; j < n; j++)
	{
		sigma = 0.0;
		ajj = 0.0;   //diagonal entry a_jj
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * x[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = (b[j] - sigma) / ajj;
	}

	//backward pass
	for (int j = n - 1; j > -1; j--)
	{
		sigma = 0.0;
		ajj = 0.0;   //diagonal entry a_jj
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * x[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = (b[j] - sigma) / ajj;
	}
}

int sgs(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double tol, const int max_iter)
{
	int ii = 0;
	double err = 1.0;
	while(err > tol && ii < max_iter)
	{
		symm_gauss_siedel_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n);

		err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
	
		#if(DEBUG)
			std::cout<<"error: "<<err<<std::endl;
		#endif
	
		ii++;
	}

	return err > tol ? -1 : ii;
}

//-------------------------------------------------------------------------------
// symmetric successive over-relaxation method
//-------------------------------------------------------------------------------
void ssor_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, const int n, const double omega)
{
	double sigma;
	double ajj;

	//forward pass
	for (int j = 0; j < n; j++)
	{
		sigma = 0.0;
		ajj = 0.0;
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * x[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = x[j] + omega * ((b[j] - sigma) / ajj - x[j]);
	}

	//backward pass
	for (int j = n - 1; j > -1; j--)
	{
		sigma = 0.0;
		ajj = 0.0;
		for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
		{
			if (csr_col_ind[k] != j)
			{
				sigma = sigma + csr_val[k] * x[csr_col_ind[k]];
			}
			else
			{
				ajj = csr_val[k];
			}
		}
		x[j] = x[j] + omega * ((b[j] - sigma) / ajj - x[j]);
	}
}

int ssor(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
         const int n, const double omega, const double tol, const int max_iter)
{
	int ii = 0;
	double err = 1.0;
	while(err > tol && ii < max_iter)
	{
		ssor_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n, omega);

		err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
		
		#if(DEBUG)
			std::cout<<"error: "<<err<<std::endl;
		#endif
	
		ii++;
	}

	return err > tol ? -1 : ii;
}
