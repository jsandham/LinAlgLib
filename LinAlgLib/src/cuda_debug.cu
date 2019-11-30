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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <time.h>
#include "../include/cuda_debug.cuh"

#include "../include/EigenValueSolvers/PowerIterationCuda.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

int CUDA_DEBUG_TEST_MATRIX(std::string filename, unsigned int nnz, unsigned int nr, unsigned int solver)
{
	int N = nnz;
	int Nr = nr;
	int Nt = 2 * N - (Nr - 1);

	// lower triangular matrix
	std::vector<int> rows;
	std::vector<int> columns;
	std::vector<float> values;

	rows.resize(Nt);
	columns.resize(Nt);
	values.resize(Nt);

	for (int i = 0; i < Nt; i++) {
		rows[i] = 0;
		columns[i] = 0;
		values[i] = 0;
	}

	std::cout << "filename: " << filename << std::endl;
	ifstream file;

	file.open(filename.c_str());

	int index = 0;
	if (file.is_open()) {

		string percent("%");
		string space(" ");
		string token;

		int currentLine = 0;

		//scan through file
		while (!file.eof())
		{
			string line;
			getline(file, line);

			if (index == Nt) { break; }

			// parse the line
			if (line.substr(0, 1).compare(percent) != 0) {
				if (currentLine > 0) {
					token = line.substr(0, line.find(space));
					int r = atoi(token.c_str()) - 1;
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					int c = atoi(token.c_str()) - 1;
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					float v = strtof(token.c_str(), NULL);

					rows[index] = r;
					columns[index] = c;
					values[index] = v;

					index++;

					if (r != c) {
						rows[index] = c;
						columns[index] = r;
						values[index] = v;

						index++;
					}
				}
				currentLine++;
			}
		}

		file.close();
	}
	else {
		std::cout << "Could not open file: " << filename << std::endl;
		return 0;
	}

	// find number of entries in each row;
	std::vector<int> entriesPerRowStartIndex;
	entriesPerRowStartIndex.resize(Nr);
	for (size_t i = 0; i < entriesPerRowStartIndex.size(); i++) {
		entriesPerRowStartIndex[i] = 0;
	}

	for (size_t i = 0; i < rows.size(); i++) {
		entriesPerRowStartIndex[rows[i] + 1]++;
	}

	for (size_t i = 0; i < entriesPerRowStartIndex.size() - 1; i++) {
		entriesPerRowStartIndex[i + 1] = entriesPerRowStartIndex[i] + entriesPerRowStartIndex[i + 1];
	}

	std::vector<int> rows_total;
	std::vector<int> cols_total;
	std::vector<float> vals_total;

	rows_total.resize(rows.size());
	cols_total.resize(columns.size());
	vals_total.resize(values.size());

	for (size_t i = 0; i < rows_total.size(); i++) {
		rows_total[i] = -1;
		cols_total[i] = -1;
		vals_total[i] = 0.0;
	}

	for (size_t i = 0; i < rows.size(); i++) {
		int startIndex = entriesPerRowStartIndex[rows[i]];
		int endIndex = entriesPerRowStartIndex[rows[i] + 1];

		// find first -1 
		for (int j = startIndex; j < endIndex; j++) {
			if (rows_total[j] == -1) {
				rows_total[j] = rows[i];
				cols_total[j] = columns[i];
				vals_total[j] = values[i];
				break;
			}
		}
	}

	// verify matrix contains a diagonal entry in every column
	for (size_t i = 0; i < entriesPerRowStartIndex.size() - 1; i++) {
		int startIndex = entriesPerRowStartIndex[i];
		int endIndex = entriesPerRowStartIndex[i + 1];

		bool diagonalFound = false;
		for (int j = startIndex; j < endIndex; j++) {
			if (rows_total[j] == cols_total[j]) {
				diagonalFound = true;
				break;
			}
		}

		if (!diagonalFound) {
			cout << "WARNING: Matrix does not contain a diagonal entry in every row/column " << std::to_string(i) << endl;
			return 0;
		}
	}

	std::vector<int> row_ptr = entriesPerRowStartIndex;

	for (int i = 0; i < row_ptr.size(); i++) {
		std::cout << row_ptr[i] << " ";// std::endl;
	}
	cout << "" << endl;
	for (int i = 0; i < cols_total.size(); i++) {
		std::cout << cols_total[i] << " ";// std::endl;
	}
	cout << "" << endl;
	for (int i = 0; i < vals_total.size(); i++) {
		std::cout << vals_total[i] << " ";// std::endl;
	}
	cout << "" << endl;

	// begin test
	srand(0);
	std::vector<float> x;
	std::vector<float> b;
	std::vector<float> temp;

	x.resize(Nr - 1);
	b.resize(Nr - 1);
	temp.resize(Nr - 1);

	//for(int i=0;i<Nr-1;i++){
	//  x[i] = 2*(rand()%Nr)/(double)Nr-1;
	//  b[i] = 1.0;
	//  //if(i<(Nr-1)/2){b[i] = 10.0;}
	//  //else{b[i] = -5.0;}
	//}

	for (int i = 0; i < Nr - 1; i++) {
		b[i] = 1.0f;
		temp[i] = 0.0f;
	}

	
	int* d_rowPtr;
	int* d_col;
	float* d_val;
	float* d_b;
	float* d_temp;
	float* d_normAb;
	float* d_alpha1;
	float* d_alpha2;

	cudaMalloc((void**)&d_rowPtr, row_ptr.size() * sizeof(int));
	cudaMalloc((void**)&d_col, cols_total.size() * sizeof(int));
	cudaMalloc((void**)&d_val, vals_total.size() * sizeof(float));
	cudaMalloc((void**)&d_b, (Nr - 1) * sizeof(float));
	cudaMalloc((void**)&d_temp, (Nr - 1) * sizeof(float));
	cudaMalloc((void**)&d_normAb, sizeof(float));
	cudaMalloc((void**)&d_alpha1, sizeof(float));
	cudaMalloc((void**)&d_alpha2, sizeof(float));

	cudaMemcpy(d_rowPtr, &row_ptr[0], row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col, &cols_total[0], cols_total.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, &vals_total[0], vals_total.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b[0], b.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp, &temp[0], temp.size() * sizeof(float), cudaMemcpyHostToDevice);


	dim3 gridSize(2, 1);
	dim3 blockSize(2, 1);
	updateEigenValue << <gridSize, blockSize >> > (d_rowPtr, d_col, d_val, d_b, d_temp, d_normAb, d_alpha1, d_alpha2, Nr - 1);

	float h_normAb = 0.0f;
	float h_alpha1 = 0.0f;
	float h_alpha2 = 0.0f;
	cudaMemcpy(&h_normAb, d_normAb, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_alpha1, d_alpha1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_alpha2, d_alpha2, sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "normAB: " << h_normAb << " alpha1: " << h_alpha1 << " alpha2: " << h_alpha2 << std::endl;

	cudaFree(d_rowPtr);
	cudaFree(d_col);
	cudaFree(d_val);
	cudaFree(d_b);
	cudaFree(d_temp);
	cudaFree(d_normAb);

	return 0;
}