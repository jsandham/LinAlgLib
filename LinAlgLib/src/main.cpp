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

// LinAlgLib.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "../include/debug.h"
#include "../include/cuda_debug.cuh"
#include "../include/LinearSolvers/SLAF.h"

int main()
{
	int erro_code;

	// tests
	for (int i = 3; i < 4; i++) {
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\test.mtx", 10, 5, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\test2.mtx", 6, 4, i);
		
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bcsstk01.mtx", 224, 49, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bcsstm02.mtx", 66, 67, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bcsstm05.mtx", 153, 154, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bcsstm22.mtx", 138, 139, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\ex5.mtx", 153, 28, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\mesh1em6.mtx", 177, 49, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\mesh2em5.mtx", 1162, 307, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\mesh3em5.mtx", 1089, 290, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\nos6.mtx", 1965, 676, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\fv1.mtx", 47434, 9605, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\fv2.mtx", 48413, 9802, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\fv3.mtx", 48413, 9802, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\ted_B.mtx", 77592, 10606, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\crystm02.mtx", 168435, 13966, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\shallow_water1.mtx", 204800, 81921, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\shallow_water2.mtx", 204800, 81921, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\thermal1.mtx", 328556, 82655, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\thermomech_TC.mtx", 406858, 102159, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\thermomech_dM.mtx", 813716, 204317, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bodyy4.mtx", 69742, 17547, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bodyy5.mtx", 73935, 18590, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\bodyy6.mtx", 77057, 19367, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\Pres_Poisson.mtx", 365313, 14823, i);






		// challenge problems
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\nos1.mtx", 627, 238, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\nos2.mtx", 2547, 958, i);
		//erro_code = DEBUG_TEST_MATRIX_AMG("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\thermomech_TK.mtx", 406858, 102159, i);

	}


	// cuda tests
	erro_code = CUDA_DEBUG_TEST_MATRIX("C:\\Users\\jsand\\Documents\\LinAlgLib\\LinAlgLib\\tests\\test.mtx", 10, 5, 0);
}
