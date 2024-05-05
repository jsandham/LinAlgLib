#include "Utility.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

struct col_val
{
	int col_ind;
	double val;
};

bool load_mtx_file(const std::string& filename, std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_ind, std::vector<double>& csr_val)
{
	std::cout << "filename: " << filename << std::endl;
	std::ifstream file;

	file.open(filename.c_str());

	std::vector<int> row_ind;
	std::vector<int> col_ind;
	std::vector<double> vals;

	int M = 0;
	int N = 0;
	int nnz = 0;

	int index = 0;
	if (file.is_open()) {

		std::string percent("%");
		std::string space(" ");
		std::string token;

		int currentLine = 0;

		//scan through file
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line);

			if (currentLine > 0 && index == nnz) { break; }

			// parse the line
			if (line.substr(0, 1).compare(percent) != 0) 
			{
				if (currentLine == 0)
				{
					std::cout << "line: " << line << std::endl;
					token = line.substr(0, line.find(space));
					M = atoi(token.c_str());
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					N = atoi(token.c_str());
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					int lower_triangular_nnz = atoi(token.c_str());

					nnz = 2 * (lower_triangular_nnz - M) + M;

					row_ind.resize(nnz);
					col_ind.resize(nnz);
					vals.resize(nnz);

					std::cout << "M: " << M << " N: " << N << " nnz: " << nnz << std::endl;
				}
				
				if (currentLine > 0) {
					token = line.substr(0, line.find(space));
					int r = atoi(token.c_str()) - 1;
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					int c = atoi(token.c_str()) - 1;
					line.erase(0, line.find(space) + space.length());
					token = line.substr(0, line.find(space));
					double v = strtod(token.c_str(), NULL);

					row_ind[index] = r;
					col_ind[index] = c;
					vals[index] = v;

					index++;

					if (r != c) {
						row_ind[index] = c;
						col_ind[index] = r;
						vals[index] = v;

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
		return false;
	}

	// find number of entries in each row;
	csr_row_ptr.resize(M + 1, 0);
	for (size_t i = 0; i < row_ind.size(); i++)
	{
		csr_row_ptr[row_ind[i] + 1]++;
	}

	for (int i = 0; i < M; i++)
	{
		csr_row_ptr[i + 1] += csr_row_ptr[i];
	}

	//std::cout << "csr_row_ptr" << std::endl;
	//for (size_t i = 0; i < csr_row_ptr.size(); i++)
	//{
	//	std::cout << csr_row_ptr[i] << " ";
	//}
	//std::cout << "" << std::endl;

	csr_col_ind.resize(nnz, -1);
	csr_val.resize(nnz, 0.0);

	for (int i = 0; i < nnz; i++)
	{
		int row_start = csr_row_ptr[row_ind[i]];
		int row_end = csr_row_ptr[row_ind[i] + 1];

		for (int j = row_start; j < row_end; j++)
		{
			if(csr_col_ind[j] == -1)
			{
				csr_col_ind[j] = col_ind[i];
				csr_val[j] = vals[i];
				break;
			}
		}
	}

	// Verify no negative 1 found in csr column indices array
	for (size_t i = 0; i < csr_col_ind.size(); i++)
	{
		if (csr_col_ind[i] == -1)
		{
			std::cout << "Error in csr_co_ind array. Negative 1 found" << std::endl;
			return false;
		}
	}

	// Sort columns and values
	for (int i = 0; i < M; i++)
	{
		int row_start = csr_row_ptr[row_ind[i]];
		int row_end = csr_row_ptr[row_ind[i] + 1];
	
		std::vector<col_val> unsorted_col_vals(row_end - row_start);
		for (int j = row_start; j < row_end; j++)
		{
			unsorted_col_vals[j - row_start].col_ind = csr_col_ind[j];
			unsorted_col_vals[j - row_start].val = csr_val[j];
		}

		std::sort(unsorted_col_vals.begin(), unsorted_col_vals.end(), [&](col_val t1, col_val t2) {
			return t1.col_ind < t2.col_ind;
			});

		for (int j = row_start; j < row_end; j++)
		{
			csr_col_ind[j] = unsorted_col_vals[j - row_start].col_ind;
			csr_val[j] = unsorted_col_vals[j - row_start].val;
		}
	}

	//std::cout << "csr_col_ind" << std::endl;
	//for (size_t i = 0; i < csr_col_ind.size(); i++)
	//{
	//	std::cout << csr_col_ind[i] << " ";
	//}
	//std::cout << "" << std::endl;

	//std::cout << "csr_val" << std::endl;
	//for (size_t i = 0; i < csr_val.size(); i++)
	//{
	//	std::cout << csr_val[i] << " ";
	//}
	//std::cout << "" << std::endl;

	return true;
}