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

#ifndef __PCGCUDA_CUH__
#define __PCGCUDA_CUH__

template<typename T, int BlockSize>
int pcg(const int* d_rowptr, const int* d_col, const T* d_val, T* d_x, const T* d_b, const T* d_res, T* d_p, T* d_z, const int n, const T tol, const int maxIter)
{
	//// jacobi preconditioner
	//double* diag = new double[n];
	//diagonal(r, c, v, diag, n);

	////res = b-A*x and initial error
	//double* res = new double[n];
	//matrixVectorProduct(r, c, v, x, res, n);
	//double err = error(r, c, v, x, b, n);
	//for (int i = 0; i < n; i++) {
	//	res[i] = b[i] - res[i];
	//}
	//if (err < tol) { return 1; }

	////create z and p vector
	//double* z = new double[n];
	//double* p = new double[n];

	////z = (M^-1)*r
	//for (int i = 0; i < n; i++) {
	//	z[i] = res[i] / diag[i];
	//}

	////p = z
	//for (int i = 0; i < n; i++) {
	//	p[i] = z[i];
	//}



	return 0;
}


#endif
