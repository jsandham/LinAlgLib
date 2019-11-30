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


#include "../include/EigenValueSolvers/PowerIteration.h"
#include "../include/LinearSolvers/SLAF.h"

#include <iostream>
#include <random>
#include <vector>

double powerIteration(const int rowptr[], const int col[], const double val[], double eigenVec[], const double tol, const int n, const int maxIter)
{
	std::vector<double> b;
	std::vector<double> Ab;
	b.resize(n);
	Ab.resize(n);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < n; i++) {
		double number = distribution(generator);
		b[i] = 1.0f;// 2.0 * number - 1.0;
	}

	double lambda = 0.0;

	// calculate A*b
	for (int i = 0; i < n; i++) {
		double s = 0.0;
		for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
			s += val[j] * b[col[j]];
		}

		Ab[i] = s;
	}

	// residual resSqr = ||A*b - lambda*b||^2
	double resSqr = 0.0;
	for (int i = 0; i < n; i++) {
		resSqr += (Ab[i] - lambda * b[i]) * ((Ab[i] - lambda * b[i]));
	}

	int iter = 0;
	while (iter < maxIter && resSqr > tol) {
		
		// calculate A*b
		for (int i = 0; i < n; i++) {
			double s = 0.0;
			for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
				s += val[j] * b[col[j]];
			}

			Ab[i] = s;
		}

		// calculate norm(A*b);
		double normAbSqr = 0.0;
		for (int i = 0; i < n; i++) {
			normAbSqr += Ab[i] * Ab[i];
		}

		// update eigenvalue
		double bAb = 0.0;
		double bb = 0.0;
		for (int i = 0; i < n; i++) {
			bAb += b[i] * Ab[i];
			bb += b[i] * b[i];
		}

		lambda = bAb / bb;

		std::cout << "eigen value: " << lambda << " normAbSqr: " << normAbSqr << " bAb: " << bAb << " bb: " << bb << " res sqr: " << resSqr << std::endl;

		// update b
		double normAb = sqrt(normAbSqr);
		for (int i = 0; i < n; i++) {
			b[i] = Ab[i] / normAb;
		}

		// update resSqr
		resSqr = 0.0;
		for (int i = 0; i < n; i++) {
			resSqr += (Ab[i] - lambda * b[i]) * ((Ab[i] - lambda * b[i]));
		}

		iter++;
	}

	// fill in eigen vector
	for (int i = 0; i < n; i++) {
		eigenVec[i] = b[i];
	}

	return lambda;
}