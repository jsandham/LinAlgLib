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

#include <iostream>
#include <linalg.h>
#include <string>

#include "test_enums.h"

namespace Testing
{
    struct Arguments
    {
        Category       category; // IterativeSolvers, Math, Primitive
        Fixture        fixture; // Jacobi, CG, SpMV, ExclusiveScan, etc
        std::string    group; // small, medium, large, etc
        std::string    filename; // bmwcra_1.mtx, shipsec1.mtx, etc
        preconditioner precond_type;
        cycle_type     cycle_type;
        smoother_type  smoother_type;
        int            presmoothing;
        int            postsmoothing;
        int            max_iters;
        int            m;
        int            n;
        double         tol;
        double         omega;

        std::string generate_test_name() const
        {
            std::string name = group;
            if(this->precond_type != preconditioner::None)
            {
                name += "_" + PreconditionerToString(this->precond_type);
            }
            if(this->cycle_type != cycle_type::None)
            {
                name += "_" + CycleTypeToString(this->cycle_type);
            }
            if(this->smoother_type != smoother_type::None)
            {
                name += "_" + SmootherTypeToString(this->smoother_type);
            }
            if(this->presmoothing >= 0)
            {
                name += "_" + std::to_string(this->presmoothing);
            }
            if(this->postsmoothing >= 0)
            {
                name += "_" + std::to_string(this->postsmoothing);
            }
            if(this->max_iters >= 0)
            {
                name += "_" + std::to_string(this->max_iters);
            }
            if(this->m >= 0)
            {
                name += "_" + std::to_string(this->m);
            }
            if(this->n >= 0)
            {
                name += "_" + std::to_string(this->n);
            }
            if(this->tol >= 0)
            {
                std::string tol_str = std::to_string(this->tol);
                for(size_t i = 0; i < tol_str.length(); i++)
                {
                    if(tol_str[i] == '.')
                    {
                        tol_str[i] = '_';
                    }
                }
                name += "_" + tol_str;
            }
            if(this->omega >= 0)
            {
                std::string omega_str = std::to_string(this->omega);
                for(size_t i = 0; i < omega_str.length(); i++)
                {
                    if(omega_str[i] == '.')
                    {
                        omega_str[i] = '_';
                    }
                }
                name += "_" + omega_str;
            }
            if(!filename.empty())
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

                name += "_" + matrix;
            }

            return name;
        }
    };
}

#endif
