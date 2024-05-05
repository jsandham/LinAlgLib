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


#include "../../include/EigenValueSolvers/PowerIteration.h"
#include "../../include/LinearSolvers/SLAF.h"

#include <iostream>
#include <random>
#include <vector>

double powerIteration(const int rowptr[], const int col[], const double val[], double eigenVec[], const double tol, const int n, const int maxIter)
{
	std::vector<double> b;
	std::vector<double> temp;
	b.resize(n);
	temp.resize(n);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < n; i++) {
		double number = distribution(generator);
		b[i] = 2.0 * number - 1.0;
	}

	double lambda = 0.0;

	// calculate A*b
	for (int i = 0; i < n; i++) {
		double s = 0.0;
		for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
			s += val[j] * b[col[j]];
		}

		temp[i] = s;
	}

	// residual resSqr = ||A*b - lambda*b||^2
	double resSqr = 0.0;
	for (int i = 0; i < n; i++) {
		resSqr += (temp[i] - lambda * b[i]) * ((temp[i] - lambda * b[i]));
	}

	int iter = 0;
	while (iter < maxIter && resSqr > tol) {
		
		// calculate A*b
		for (int i = 0; i < n; i++) {
			double s = 0.0;
			for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
				s += val[j] * b[col[j]];
			}

			temp[i] = s;
		}

		// calculate norm(A*b);
		double normAb = 0.0;
		for (int i = 0; i < n; i++) {
			normAb += temp[i] * temp[i];
		}

		// update eigenvalue
		double alpha1 = 0.0;
		double alpha2 = 0.0;
		for (int i = 0; i < n; i++) {
			alpha1 += b[i] * temp[i];
			alpha2 += b[i] * b[i];
		}

		lambda = alpha1 / alpha2;

		std::cout << "eigen value: " << lambda << " normAb: " << normAb << " res sqr: " << resSqr << std::endl;

		// update b
		normAb = sqrt(normAb);
		for (int i = 0; i < n; i++) {
			b[i] = temp[i] / normAb;
		}

		// update resSqr
		resSqr = 0.0;
		for (int i = 0; i < n; i++) {
			resSqr += (temp[i] - lambda * b[i]) * ((temp[i] - lambda * b[i]));
		}

		iter++;
	}

	// fill in eigen vector
	for (int i = 0; i < n; i++) {
		eigenVec[i] = b[i];
	}

	return lambda;
}