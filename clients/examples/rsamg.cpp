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

#include <vector>
#include <iostream>

#include "linalg.h"
#include "utility.h"

int main()
{
	std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	std::vector<double> csr_val;
	load_mtx_file("../clients/matrices/fv1.mtx", csr_row_ptr, csr_col_ind, csr_val);

	int m = csr_row_ptr.size() - 1;

	// Solution vector
	std::vector<double> x(m, 0.0);

	// Righthand side vector
	std::vector<double> b(m, 1.0);

	amg(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), m, 0.5, 0.00001);

	// Print solution
	//std::cout << "x" << std::endl;
	//for (size_t i = 0; i < x.size(); i++)
	//{
	//	std::cout << x[i] << " ";
	//}
	//std::cout << "" << std::endl;

	return 0;
}