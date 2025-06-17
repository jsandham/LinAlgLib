//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
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

#ifndef TEST_ARGUMENTS_H__
#define TEST_ARGUMENTS_H__

#include <string>
#include <iostream>
#include <linalg.h>

#include "test_enums.h"

namespace Testing
{
    struct Arguments
    {
        std::string filename;
        Solver solver;
        preconditioner precond;
        linalg::Cycle cycle;
        linalg::Smoother smoother;
        int presmoothing;
        int postsmoothing;
        int max_iters;
        double tol;
        double omega;

        std::string generate_test_name() const
        {
            size_t index = 0;
            for(size_t i = 0; i < filename.length(); i++)
            {
                if(filename[i] == '/')
                {
                    index = i;
                }
            }

            std::string matrix(filename.length() - index - 1, '0');
            for(size_t i = index + 1; i < filename.length(); i++)
            {
                if(filename[i] == '.')
                {
                    matrix[i - index - 1] = '_';
                }
                else
                {
                    matrix[i - index - 1] = filename[i];
                }
            }

            std::string tol_str = std::to_string(this->tol);
            for(size_t i = 0; i < tol_str.length(); i++)
            {
                if(tol_str[i] == '.')
                {
                    tol_str[i] = '_';
                }
            }

            std::string omega_str = std::to_string(this->omega);
            for(size_t i = 0; i < omega_str.length(); i++)
            {
                if(omega_str[i] == '.')
                {
                    omega_str[i] = '_';
                }
            }
            
            std::string name = SolverToString(this->solver) + "_" 
                             + PreconditionerToString(this->precond) + "_"
                             + CycleToString(this->cycle) + "_"
                             + SmootherToString(this->smoother) + "_"
                             + std::to_string(this->presmoothing) + "_"
                             + std::to_string(this->postsmoothing) + "_"
                             + std::to_string(this->max_iters) + "_"
                             + tol_str + "_"
                             + omega_str + "_"
                             + matrix;

            return name;
        }
    };
}

#endif
