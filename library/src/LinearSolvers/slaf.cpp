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
#include"../../include/LinearSolvers/slaf.h"
#include"math.h"

//********************************************************************************
//
// Sparse linear algebra functions
//
//********************************************************************************


//-------------------------------------------------------------------------------
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void matrixVectorProduct(const int r[], const int c[], const double v[],
                         const double x[], double y[], const int n)
{
	for(int i = 0; i < n; i++){
		double s = 0.0;
		for(int j = r[i]; j < r[i + 1]; j++)
			s += v[j] * x[c[j]];
		y[i] = s;
	}
}

//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
double dotProduct(const double x[], const double y[], const int n)
{
	double dot_prod = 0.0;
	for(int i = 0; i < n; i++){
		dot_prod = dot_prod + x[i] * y[i];
	}

	return dot_prod;
}

//-------------------------------------------------------------------------------
// diagonal d = diag(A)
//-------------------------------------------------------------------------------
void diagonal(const int r[], const int c[], const double v[], double d[], const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = r[i]; j < r[i + 1]; j++) {
			if (c[j] == i) {
				d[i] = v[j];
				break;
			}
		}
	}
}

//-------------------------------------------------------------------------------
// solve Lx = b where L is a lower triangular sparse matrix
//-------------------------------------------------------------------------------
void forwardSolve(const int r[], const int c[], const double v[], const double b[], double x[], const int n)
{
	for (int i = 0; i < n; i++) {
		x[i] = b[i];
		for (int j = r[i]; j < r[i + 1]; j++) {
			if (c[j] < i) {
				x[i] -= v[j] * x[c[j]];
			}
		}
		x[i] /= v[r[i + 1] - 1];
	}
}

//-------------------------------------------------------------------------------
// solve Ux = b where U is a upper triangular sparse matrix
//-------------------------------------------------------------------------------
void backwardSolve(const int r[], const int c[], const double v[], const double b[], double x[], const int n)
{
	for (int i = n - 1; i >= 0; i--) {
		x[i] = b[i];
		for (int j = r[i + 1] - 1; j >= r[i]; j--) {
			if (c[j] > i) {
				x[i] -= v[j] * x[c[j]];
			}
		}

		x[i] /= v[r[i]];
	}
}




//-------------------------------------------------------------------------------
// error e = |b-A*x|
//-------------------------------------------------------------------------------
double error(const int r[], const int c[], const double v[], const double x[],
             const double b[], const int n)
{
  double e = 0.0;
  for(int j=0;j<n;j++){
    double s = 0.0;
    for(int i=r[j];i<r[j+1];i++)
      s += v[i]*x[c[i]];
    e = e + (b[j] - s)*(b[j] - s);
  }

  return sqrt(e);
}


//-------------------------------------------------------------------------------
// error e = |b-A*x| stops calculating error if error goes above tolerance 
//-------------------------------------------------------------------------------
double fast_error(const int r[], const int c[], const double v[], const double x[],
                  const double b[], const int n, const double tol)
{
  int j = 0;
  double e = 0.0;
  while(e<tol && j<n){
    double s = 0.0;
    for(int i=r[j];i<r[j+1];i++)
      s += v[i]*x[c[i]];
    e = e + (b[j] - s)*(b[j] - s);
    j++;
  }

  return sqrt(e);
}


//-------------------------------------------------------------------------------
// print matrix 
//-------------------------------------------------------------------------------
void printMatrix(const int r[], const int c[], const double v[], const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = r[i]; j < r[i + 1]; j++) {
			std::cout << v[j] << " ";
		}
		std::cout << "" << std::endl;
	}
}

