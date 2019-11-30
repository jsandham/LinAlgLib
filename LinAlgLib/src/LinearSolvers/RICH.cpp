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
#include"../../include/LinearSolvers/RICH.h"
#include"../../include/LinearSolvers/SLAF.h"
#include"math.h"

//********************************************************************************
//
// Richardson Iteration
//
//********************************************************************************

#define DEBUG 1


//-------------------------------------------------------------------------------
// richardson method
//-------------------------------------------------------------------------------
int rich(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double theta, const double tol, const int max_iter)
{
	//res = b-A*x and initial error
	double *res = new double[n];
	matrixVectorProduct(r,c,v,x,res,n);
	for(int i=0;i<n;i++){res[i] = b[i] - res[i];}
	double err = error(r,c,v,x,b,n);
	if(err<tol){return 1;}
  

	int iter = 0, inner_iter = 0;
	while(iter<max_iter && err>tol){
		//find res = A*x
		matrixVectorProduct(r,c,v,x,res,n);

		//update approximation
		for(int i=0;i<n;i++){
			x[i] = x[i] + theta*(b[i] - res[i]);
		}

		//calculate error
		if(inner_iter==40){
			err = error(r, c, v, x, b, n);
			inner_iter = 0;
			#if(DEBUG)
				std::cout<<"error: "<<err<<std::endl;
			#endif
		}
		iter++;
		inner_iter++;   
	}

	delete[] res;

	return iter;
}
