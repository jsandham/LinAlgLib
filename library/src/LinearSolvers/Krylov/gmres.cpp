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


#include<iostream>
#include <vector>
#include"../../../include/LinearSolvers/Krylov/gmres.h"
#include"../../../include/LinearSolvers/slaf.h"
#include"math.h"


//****************************************************************************
//
// Generalised Minimum Residual
//
//****************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// generalised minimum residual
//-------------------------------------------------------------------------------
int gmres(const int*csr_row_ptr, const int*csr_col_ind, const double* csr_val, double* x, const double* b,
	const int n, const int restart, const double tol, const int max_iter)
{
	//res = b-A*x and initial error
	std::vector<double> res;
	res.resize(n);

	//create H and Q matrices (which are dense and stored as vectors columnwise) 
	std::vector<double> H;
	std::vector<double> R;
	std::vector<double> Q;
	std::vector<double> cg;
	std::vector<double> sg;
	std::vector<double> q;
	std::vector<double> v;
	H.resize(restart * (restart - 1));
	R.resize(restart * (restart - 1));
	Q.resize(n * restart);
	cg.resize(restart - 1);
	sg.resize(restart - 1);
	q.resize(n);
	v.resize(n);

	//initialize H and Q matrices to zero
	for (int i = 0; i < restart * (restart - 1); i++) 
	{
		H[i] = 0.0;
		R[i] = 0.0;
	}
	for (int i = 0; i < n * restart; i++) 
	{ 
		Q[i] = 0.0; 
	}

	matrixVectorProduct(csr_row_ptr, csr_col_ind, csr_val, x, &res[0], n);
	for (int i = 0; i < n; i++) 
	{
		res[i] = b[i] - res[i];
	}

	double res_norm = sqrt(dotProduct(&res[0], &res[0], n));
	double b_norm = sqrt(dotProduct(b, b, n));
	double error = res_norm / b_norm;
	for (int i = 0; i < n; i++) {
		q[i] = res[i] / res_norm;
	}
	for (int i = 0; i < n; i++) {
		Q[i] = q[i];
	}

	std::vector<double> beta;
	beta.resize(n);
	for (int i = 0; i < n; i++) {
		beta[i] = 0.0;
	}
	beta[0] = res_norm;

	//gmres 
	int iter = 0, k = 0;
	while (iter < n - 1 && error > tol) {
		iter++;
		k++;

		std::cout << "k: " << k << std::endl;

		//Arnoldi iteration
		matrixVectorProduct(csr_row_ptr, csr_col_ind, csr_val, &Q[(k-1)*n], &v[0], n);
		for (int i = 0; i < k; i++) {
			H[i + (k - 1) * restart] = dotProduct(&Q[i * n], &v[0], n);
			for (int j = 0; j < n; j++) {
				v[j] = v[j] - H[i + (k - 1) * restart] * Q[j + i * n];
			}
		}

		double vv = dotProduct(&v[0], &v[0], n);
		H[k + (k - 1) * restart] = sqrt(vv);
		if (vv < 10e-12) {
			for (int i = 0; i < n; i++) {
				q[i] = 0.0;
			}
		}
		else {
			for (int i = 0; i < n; i++) {
				q[i] = v[i] / H[k + (k - 1) * restart];
			}
		}
		for (int i = 0; i < n; i++) {
			Q[i + k * n] = q[i];
		}

		//solve least squares problem h(1:k+1,1:k)*y=sqrt(b'*b)*eye(k+1,1)
		//since H is hessenberg, use givens rotations
		for (int i = 0; i < k + 1; i++) {
			R[i + (k - 1) * restart] = H[i + (k - 1) * restart];
		}

		// printing
		/*std::cout << "H" << std::endl;
		for (int i = 0; i < restart; i++) {
			for (int j = 0; j < (restart - 1); j++) {
				std::cout << H[i + j * restart] << " ";
			}
			std::cout << "" << std::endl;
		}
		std::cout << "" << std::endl;

		std::cout << "Q" << std::endl;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < restart; j++) {
				std::cout << Q[i + j * n] << " ";
			}
			std::cout << "" << std::endl;
		}
		std::cout << "" << std::endl;

		std::cout << "R" << std::endl;
		for (int i = 0; i < restart; i++) {
			for (int j = 0; j < (restart - 1); j++) {
				std::cout << R[i + j * restart] << " ";
			}
			std::cout << "" << std::endl;
		}
		std::cout << "" << std::endl;*/

		// Apply previous cached givens rotations
		// Givens 2 by 2 rotation matrix: 
		//  G = [g11,g12  =  [c s
		//       g21,g22]    -s c]
		for (int i = 0; i < k - 1; i++) {
			double g11 = cg[i], g12 = sg[i], g21 = -sg[i], g22 = cg[i];

			double temp1 = R[i + (k - 1) * restart];
			double temp2 = R[i + 1 + (k - 1) * restart];

			//std::cout << "first: " << temp1 << " second: " << temp2 << " g11: " << g11 << " g12: " << g12 << " g21: " << g21 << " g22: " << g22 << std::endl;
			R[i + (k - 1) * restart] = g11 * temp1 + g12 * temp2;
			R[i + 1 + (k - 1) * restart] = g21 * temp1 + g22 * temp2;
		}

		// calculate new givens rotation
		double xi = R[k - 1 + (k - 1) * restart], xj = R[k + (k - 1) * restart];
		double c, s;
		if (xi < 10e-12 && xj < 10e-12) {
			c = 0.0; s = 0.0;
		}
		else {
			c = xi / sqrt(xi * xi + xj * xj);
			s = xj / sqrt(xi * xi + xj * xj);
		}
		cg[k - 1] = c;
		sg[k - 1] = s;

		// apply newest givens rotation to eliminate off diagonal entry
		double g11 = c, g12 = s, g21 = -s, g22 = c;
		double temp1 = R[k - 1 + (k - 1) * restart];
		double temp2 = R[k + (k - 1) * restart];

		//std::cout << "first: " << temp1 << " second: " << temp2 << " g11: " << g11 << " g12: " << g12 << " g21: " << g21 << " g22: " << g22 << std::endl;
		R[k - 1 + (k - 1) * restart] = g11 * temp1 + g12 * temp2;
		R[k + (k - 1) * restart] = g21 * temp1 + g22 * temp2;

		// update residual vector
		beta[k] = g21 * beta[k - 1];
		beta[k - 1] = g11 * beta[k - 1];
		error = abs(beta[k]) / b_norm;

		/*if (error <= tol)
			break;*/

		//backward solve
		std::vector<double> y;
		y.resize(k);
		for (int i = k - 1; i > -1; i--) {
			if (i + 1 > k - 1) {
				/*y[i] = beta[i] / R[i + (k + 1) * i];*/
				y[i] = beta[i] / R[i + restart * i];
			}
			else {
				for (int j = i + 1; j < k; j++) {
					y[i] = (beta[i] - R[i + restart * j] * y[j]) / R[i + restart * i];
				}
			}
		}

		/*std::cout << "y" << std::endl;
		for (int i = 0; i < k; i++) {
			std::cout << y[i] << std::endl;
		}*/

		// update solution vector
		for (int i = 0; i < k; i++) 
		{
			for (int j = 0; j < k; j++) 
			{
				x[i] = x[i] + Q[j * n + i] * y[j];
			}
		}

#if(DEBUG)
		std::cout << "error: " << error << std::endl;
#endif
	}

	return iter;
}

















































////-------------------------------------------------------------------------------
//// generalised minimum residual
////-------------------------------------------------------------------------------
//int gmres(const int row[], const int col[], const double val[], double x[], const double b[], 
//          const int n, const int restart, const double tol, const int max_iter)
//{
//	//res = b-A*x and initial error
//	std::vector<double> res;
//	res.resize(n);
//	matrixVectorProduct(row, col, val, x, &res[0], n);
//	double err = error(row, col, val, x, b, n);
//	for(int i = 0; i < n; i++){
//		res[i] = b[i] - res[i];
//	}
//	if(err < tol){return 1;}
//
//	//create q and v array
//	std::vector<double> q;
//	std::vector<double> v;
//	q.resize(n);
//	v.resize(n);
//  
//	//create H and Q matrices (which are dense and stored as vectors columnwise) 
//	std::vector<double> H;
//	std::vector<double> R;
//	std::vector<double> Q;
//	std::vector<double> cg;
//	std::vector<double> sg;
//	H.resize(restart * (restart - 1));
//	R.resize(restart * (restart - 1));
//	Q.resize(n * restart);
//	cg.resize(restart - 1);
//	sg.resize(restart - 1);
//
//	/*std::vector<double> x0;
//	x0.resize(n);
//	for (int i = 0; i < n; i++) {
//		x0[i] = x[i];
//	}*/
//
//
//	//initialize H and Q matrices to zero
//	for(int i = 0; i < restart * (restart - 1); i++){
//		H[i] = 0.0;
//		R[i] = 0.0;
//	}
//	for(int i = 0; i < n * restart; i++){Q[i] = 0.0;}
//
//	double bb = dotProduct(b, b, n);
//	for (int i = 0; i < n; i++) { 
//		q[i] = b[i] / sqrt(bb); 
//	}
//	for(int i = 0; i < n; i++){
//		Q[i] = q[i];
//	}
//
//	std::vector<double> beta;
//	beta.resize(n);
//	for (int i = 0; i < n; i++) {
//		beta[i] = 0.0;
//	}
//	beta[0] = sqrt(bb);
//
//	//gmres 
//	int iter = 0, k = 0;
//	while(iter < max_iter && err > tol){
//		iter++;
//
//		//restart
//		while(k < restart - 1){
//			k++;
//			std::cout << "k: " << k << std::endl;
//
//			//Arnoldi iteration
//			matrixVectorProduct(row, col, val, &q[0], &v[0], n);
//			for(int i = 0; i < k; i++){
//				H[i + (k-1) * restart] = dotProduct(&Q[i*n], &v[0], n); 
//				for(int j = 0; j < n; j++){
//					v[j] = v[j] - H[i + (k-1) * restart] * Q[j + i * n];
//				}
//			}
//
//			double vv = dotProduct(&v[0], &v[0], n);
//			H[k + (k-1) * restart] = sqrt(vv);
//			if(vv < 10e-12){
//				for(int i = 0; i < n; i++){
//					q[i] = 0.0;
//				}
//			}
//			else{
//				for(int i = 0; i < n; i++){
//					q[i] = v[i] / H[k + (k-1) * restart];
//				}
//			}
//			for(int i = 0; i < n; i++){
//				Q[i + k * n] = q[i];
//			}
//
//
//
//
//			std::cout << "H" << std::endl;
//			for (int i = 0; i < restart; i++) {
//				for (int j = 0; j < (restart - 1); j++) {
//					std::cout << H[i + j * restart] << " ";
//				}
//				std::cout << "" << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//			std::cout << "v" << std::endl;
//			for (int i = 0; i < n; i++) {
//				std::cout << v[i] << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//			std::cout << "Q" << std::endl;
//			for (int i = 0; i < n; i++) {
//				for (int j = 0; j < restart; j++) {
//					std::cout << Q[i + j * n] << " ";
//				}
//				std::cout << "" << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//
//			//solve least squares problem h(1:k+1,1:k)*y=sqrt(b'*b)*eye(k+1,1)
//			//since H is hessenberg, use givens rotations
//			for (int i = 0; i < k + 1; i++) {
//				R[i + (k - 1) * restart] = H[i + (k - 1) * restart];
//			}
//
//			std::cout << "R" << std::endl;
//			for (int i = 0; i < restart; i++) {
//				for (int j = 0; j < (restart - 1); j++) {
//					std::cout << R[i + j * restart] << " ";
//				}
//				std::cout << "" << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//			// Apply previous cached givens rotations
//			// Givens 2 by 2 rotation matrix: 
//			//  G = [g11,g12  =  [c s
//			//       g21,g22]    -s c]
//			for (int i = 0; i < k - 1; i++) {
//				double g11 = cg[i], g12 = sg[i], g21 = -sg[i], g22 = cg[i];
//
//				double temp1 = R[i + (k - 1) * restart];
//				double temp2 = R[i + 1 + (k - 1) * restart];
//
//				std::cout << "first: " << temp1 << " second: " << temp2 << " g11: " << g11 << " g12: " << g12 << " g21: " << g21 << " g22: " << g22 << std::endl;
//				R[i + (k - 1) * restart] = g11 * temp1 + g12 * temp2;
//				R[i + 1 + (k - 1) * restart] = g21 * temp1 + g22 * temp2;
//			}
//
//			// calculate new givens rotation
//			double xi = R[k-1 + (k-1) * restart], xj = R[k + (k-1) * restart];
//			double c, s;
//			if (xi < 10e-12 && xj < 10e-12) {
//				c = 0.0; s = 0.0;
//			}
//			else {
//				c = xi / sqrt(xi * xi + xj * xj);
//				s = xj / sqrt(xi * xi + xj * xj);
//			}
//			cg[k - 1] = c;
//			sg[k - 1] = s;
//
//			// apply newest givens rotation to eliminate off diagonal entry
//			double g11 = c, g12 = s, g21 = -s, g22 = c;
//			double temp1 = R[k-1 + (k - 1) * restart];
//			double temp2 = R[k + (k - 1) * restart];
//
//			std::cout << "first: " << temp1 << " second: " << temp2 << " g11: " << g11 << " g12: " << g12 << " g21: " << g21 << " g22: " << g22 << std::endl;
//			R[k - 1 + (k - 1) * restart] = g11 * temp1 + g12 * temp2;
//			R[k + (k - 1) * restart] = g21 * temp1 + g22 * temp2;
//
//			std::cout << "should be zero: " << g21 * temp1 + g22 * temp2 << std::endl;
//
//
//			std::cout << "R" << std::endl;
//			for (int i = 0; i < restart; i++) {
//				for (int j = 0; j < (restart - 1); j++) {
//					std::cout << R[i + j * restart] << " ";
//				}
//				std::cout << "" << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//			// update residual vector
//			beta[k] = g21 * beta[k - 1];
//			beta[k - 1] = g11 * beta[k - 1];
//			err = abs(beta[k]) / sqrt(bb);
//
//			if (err <= tol)
//				break;
//
//			std::cout << "beta" << std::endl;
//			for (int i = 0; i < n; i++) {
//				std::cout << beta[i] << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//			//backward solve
//			std::vector<double> y;
//			y.resize(k);
//			for(int i = k - 1; i > -1; i--){
//				if(i + 1 > k-1){
//					/*y[i] = beta[i] / R[i + (k + 1) * i];*/
//					y[i] = beta[i] / R[i + restart * i];
//				}
//				else{
//					for(int j = i + 1; j < k; j++){
//						y[i] = (beta[i] - R[i + restart * j] * y[j]) / R[i + restart * i];
//					}
//				}
//			}
//
//			std::cout << "y" << std::endl;
//			for (int i = 0; i < k; i++) {
//				std::cout << y[i] << std::endl;
//			}
//
//			// update solution vector
//			for (int i = 0; i < k; i++) {
//				for (int j = 0; j < k; j++) {
//					x[i] = x[i] + Q[j * n + i] * y[j];
//				}
//			}
//
//			std::cout << "x" << std::endl;
//			for (int i = 0; i < k; i++) {
//				std::cout << x[i] << std::endl;
//			}
//			std::cout << "" << std::endl;
//
//
//
//#if(DEBUG)
//			std::cout << "error: " << err << std::endl;
//#endif
//      
//		}
//
//		k = 0;
//	}
//
//	return iter;
//}


























































//	// update residual vector
//	if (i == k - 1) {
//		beta[k] = -g21 * beta[k - 1];
//		beta[k - 1] = g11 * beta[k - 1];
//		err = abs(beta[k]) / sqrt(bb);
//	}

//for (int i = 0; i < k; i++) {
//	// Apply Givens 2 by 2 rotation matrix: 
//	//  G = [g11,g12  =  [c s
//	//       g21,g22]    -s c]
//	double g11 = cg[i], g12 = sg[i], g21 = -sg[i], g22 = cg[i];
//	//double g11 = 0.0, g12 = 0.0, g21 = 0.0, g22 = 0.0;
//	///*double xi = H[i + i * restart], xj = H[i + 1 + i * restart];*/
//	//double xi = R[i + i * restart], xj = R[i + 1 + i * restart];
//	//double c, s;
//	//if (xi < 10e-12 && xj < 10e-12) {
//	//	c = 0.0; s = 0.0;
//	//}
//	//else {
//	//	c = xi / sqrt(xi * xi + xj * xj);
//	//	s = -xj / sqrt(xi * xi + xj * xj);
//	//}
//	//g11 = c; g22 = c; g12 = -s; g21 = s;

//	// apply givens rotation to get upper triangular R matrix
//	std::cout << "first: " << R[i + i * restart] << " second: " << R[i + 1 + i * restart] << " g11: " << g11 << " g12: " << g12 << " g21: " << g21 << " g22: " << g22 << std::endl;
//	/*R[i + i * (k + 1)] = g11 * R[i + i * restart] + g12 * R[i + 1 + i * restart];
//	R[i + 1 + i * (k + 1)] = g21 * R[i + i * restart] + g22 * R[i + 1 + i * restart];*/
//	double temp1 = R[i + i * restart];
//	double temp2 = R[i + 1 + i * restart];
//	R[i + i * (k + 1)] = g11 * temp1 + g12 * temp2;
//	R[i + 1 + i * (k + 1)] = g21 * temp1 + g22 * temp2;

//	// update residual vector
//	if (i == k - 1) {
//		beta[k] = -g21 * beta[k - 1];
//		beta[k - 1] = g11 * beta[k - 1];
//		err = abs(beta[k]) / sqrt(bb);
//	}
//}