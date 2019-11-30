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

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <string>


#define DEBUG 1
#define TIMING 0

#define DEBUG_PRINT(stream, statement) \
	do { if(DEBUG) (stream) << "DEBUG: "<< __FILE__<<"("<<__LINE__<<") " << (statement) << std::endl;} while(0)

#define TIMING_PRINT(stream, statement, time) \
	do { if(TIMING) (stream) << "TIMING: "<<__FILE__<<"("<<__LINE__<<") " << (statement) << (time) << std::endl;} while(0)

int DEBUG_TEST_MATRIX_AMG(std::string filename, unsigned int nnz, unsigned int nr, unsigned int solver);


#endif