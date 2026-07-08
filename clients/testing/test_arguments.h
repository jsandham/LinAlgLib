//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

namespace testing
{
    struct Arguments
    {
        testing::category          category; // iterative_solvers, math, primitive
        testing::fixture           fixture; // Jacobi, CG, SpMV, ExclusiveScan, etc
        std::string                group; // small, medium, large, etc
        std::string                filename; // bmwcra_1.mtx, shipsec1.mtx, etc
        testing::backend           backend; // CPU, GPU
        testing::uplo              uplo; // Lower, Upper
        testing::preconditioner    precond_type;
        testing::cycle_type        cycle_type;
        testing::smoother_type     smoother_type;
        testing::pivoting_strategy pivoting_strategy;
        int                        presmoothing;
        int                        postsmoothing;
        int                        max_iters;
        int                        m;
        int                        n;
        double                     tol;
        double                     omega;

        std::string generate_test_name() const
        {
            std::string name = group;
            if(this->backend != backend::CPU)
            {
                name += "_" + backend_to_string(this->backend);
            }
            if(this->uplo != uplo::Lower)
            {
                name += "_" + uplo_to_string(this->uplo);
            }
            if(this->precond_type != preconditioner::None)
            {
                name += "_" + preconditioner_to_string(this->precond_type);
            }
            if(this->cycle_type != cycle_type::None)
            {
                name += "_" + cycle_type_to_string(this->cycle_type);
            }
            if(this->smoother_type != smoother_type::None)
            {
                name += "_" + smoother_type_to_string(this->smoother_type);
            }
            if(this->pivoting_strategy != pivoting_strategy::None)
            {
                name += "_" + pivoting_strategy_to_string(this->pivoting_strategy);
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
