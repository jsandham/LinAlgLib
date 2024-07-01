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

#include <vector>
#include <iostream>

#include "linalg.h"
#include "utility.h"

int main()
{
	std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	std::vector<double> csr_val;
	load_mtx_file("../clients/matrices/mesh1em6.mtx", csr_row_ptr, csr_col_ind, csr_val);

	int m = (int)csr_row_ptr.size() - 1;
	int n = m;
	int nnz = (int)csr_val.size();

	std::cout << "A" << std::endl;
	for (int i = 0; i < m; i++)
	{
		int start = csr_row_ptr[i];
		int end = csr_row_ptr[i + 1];

		std::vector<double> temp(n, 0);
		for (int j = start; j < end; j++)
		{
			temp[csr_col_ind[j]] = csr_val[j];
		}

		for (size_t j = 0; j < temp.size(); j++)
		{
			std::cout << temp[j] << " ";
		}
		std::cout << "" << std::endl;
	}
	std::cout << "" << std::endl;


	// Solution vector
	std::vector<double> x(m, 0.0);

	// Righthand side vector
	std::vector<double> b(m, 1.0);

	saamg_heirarchy hierachy;
	saamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 2, 0.5, hierachy);


	//amg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.5, 0.00001);

	// Print solution
	//std::cout << "x" << std::endl;
	//for (size_t i = 0; i < x.size(); i++)
	//{
	//	std::cout << x[i] << " ";
	//}
	//std::cout << "" << std::endl;

	return 0;
}