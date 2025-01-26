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

#ifndef TEST_H__
#define TEST_H__

#include <string>

#include <gtest/gtest.h>

#include "test_classical.h"
#include "test_krylov.h"
#include "test_amg.h"

std::string classical_matrix_files[] = {"bcsstm02.mtx",
                                     "bcsstm05.mtx",
                                     "bcsstm22.mtx",
                                     "bodyy4.mtx",
                                     "bodyy5.mtx",
                                     "bodyy6.mtx",
                                     "nos1.mtx",
                                     "nos6.mtx",
                                     "ex5.mtx",
                                     "crystm02.mtx",
                                     "fv1.mtx",
                                     "fv2.mtx",
                                     "fv3.mtx",
                                     "mesh1em6.mtx",
                                     "mesh2em5.mtx",
                                     "mesh3em5.mtx",
                                     "ted_B.mtx",
                                     "test.mtx",
                                     "shallow_water1.mtx",
                                     "shallow_water2.mtx"};
Testing::ClassicalSolver classical_solvers[] = {Testing::ClassicalSolver::Jacobi, 
                                                Testing::ClassicalSolver::GaussSeidel, 
                                                Testing::ClassicalSolver::SOR,
                                                Testing::ClassicalSolver::SymmGaussSeidel,
                                                Testing::ClassicalSolver::SSOR};

class classical_parameters : public testing::TestWithParam<std::tuple<Testing::ClassicalSolver, std::string>>
{
  protected:
    classical_parameters(){}
    virtual ~classical_parameters(){}
    virtual void SetUp(){}
    virtual void TearDown(){}
};

TEST_P(classical_parameters, classical_test)
{
    Testing::ClassicalSolver solver = std::get<0>(GetParam());
    std::string filename = "../matrices/" + std::get<1>(GetParam());
    EXPECT_EQ(1, Testing::test_classical(solver, filename));
}

INSTANTIATE_TEST_CASE_P(classical, 
                        classical_parameters, 
                        testing::Combine(testing::ValuesIn(classical_solvers), 
                                         testing::ValuesIn(classical_matrix_files)), 
                        [](const testing::TestParamInfo<classical_parameters::ParamType>& info) 
                        {
                          std::string solver = Testing::ClassicalSolverToString(std::get<0>(info.param));
                          std::string filename = std::get<1>(info.param);
                          for(size_t i = 0; i < filename.length(); i++)
                          {
                            if(filename[i] == '.')
                            {
                              filename[i] = '_';
                            }
                          }
                          return solver + "_" + filename;
                        });

std::string krylov_matrix_files[] = {"bcsstm02.mtx",
                                     "bcsstm05.mtx",
                                     "bcsstm22.mtx",
                                     "bodyy4.mtx",
                                     //"bodyy5.mtx",
                                     //"bodyy6.mtx",
                                     "nos1.mtx",
                                     //"nos6.mtx",
                                     "ex5.mtx",
                                     "crystm02.mtx",
                                     "fv1.mtx",
                                     "fv2.mtx",
                                     "fv3.mtx",
                                     "mesh1em6.mtx",
                                     "mesh2em5.mtx",
                                     "mesh3em5.mtx",
                                     "ted_B.mtx",
                                     "test.mtx",
                                     "shallow_water1.mtx",
                                     "shallow_water2.mtx"};
Testing::KrylovSolver krylov_solvers[] = {Testing::KrylovSolver::CG, Testing::KrylovSolver::BICGSTAB};
Testing::Preconditioner krylov_precondioners[] = {Testing::Preconditioner::None, Testing::Preconditioner::Jacobi, Testing::Preconditioner::ILU, Testing::Preconditioner::IC};

class krylov_parameters : public testing::TestWithParam<std::tuple<Testing::KrylovSolver, Testing::Preconditioner, std::string>>
{
  protected:
    krylov_parameters(){}
    virtual ~krylov_parameters(){}
    virtual void SetUp(){}
    virtual void TearDown(){}
};

TEST_P(krylov_parameters, krylov_test)
{
    Testing::KrylovSolver solver = std::get<0>(GetParam());
     Testing::Preconditioner preconditioner = std::get<1>(GetParam());
    std::string filename = "../matrices/" + std::get<2>(GetParam());
    EXPECT_EQ(1, Testing::test_krylov(solver, preconditioner, filename));
}

INSTANTIATE_TEST_CASE_P(krylov, 
                        krylov_parameters, 
                        testing::Combine(testing::ValuesIn(krylov_solvers),
                                         testing::ValuesIn(krylov_precondioners), 
                                         testing::ValuesIn(krylov_matrix_files)), 
                        [](const testing::TestParamInfo<krylov_parameters::ParamType>& info) 
                        {
                          std::string solver = Testing::KrylovSolverToString(std::get<0>(info.param));
                          std::string preconditioner = Testing::PreconditionerToString(std::get<1>(info.param));
                          std::string filename = std::get<2>(info.param);
                          for(size_t i = 0; i < filename.length(); i++)
                          {
                            if(filename[i] == '.')
                            {
                              filename[i] = '_';
                            }
                          }
                          return solver + "_" + preconditioner + "_" + filename;
                        }
                        );

std::string amg_matrix_files[] = {"bcsstm02.mtx",
                                     "bcsstm05.mtx",
                                     "bcsstm22.mtx",
                                     "bodyy4.mtx",
                                     "bodyy5.mtx",
                                     "bodyy6.mtx",
                                     "nos1.mtx",
                                     "nos6.mtx",
                                     "ex5.mtx",
                                     "crystm02.mtx",
                                     "fv1.mtx",
                                     "fv2.mtx",
                                     "fv3.mtx",
                                     "mesh1em6.mtx",
                                     "mesh2em5.mtx",
                                     "mesh3em5.mtx",
                                     "ted_B.mtx",
                                     "test.mtx",
                                     "shallow_water1.mtx",
                                     "shallow_water2.mtx"};
Testing::AMGSolver amg_solvers[] = {Testing::AMGSolver::SAAMG};
int amg_presmoothing_iterations[] = {2};
int amg_postsmoothing_iterations[] = {4};
Cycle cycle_types[] = {Cycle::Fcycle, Cycle::Vcycle, Cycle::Wcycle};
Smoother smoother_type[] = {Smoother::Jacobi, Smoother::Gauss_Siedel, Smoother::SOR};

class amg_parameters : public testing::TestWithParam<std::tuple<Testing::AMGSolver, int, int, Cycle, Smoother, std::string>>
{
  protected:
    amg_parameters()
    {
    }
    virtual ~amg_parameters()
    {
    }
    virtual void SetUp()
    {
    }
    virtual void TearDown()
    {
    }
};

TEST_P(amg_parameters, amg_test)
{
    Testing::AMGSolver solver = std::get<0>(GetParam());
    int presmoothing = std::get<1>(GetParam());
    int postsmoothing = std::get<2>(GetParam());
    Cycle cycle = std::get<3>(GetParam());
    Smoother smoother = std::get<4>(GetParam());
    std::string filename = "../matrices/" + std::get<5>(GetParam());
    EXPECT_EQ(1, Testing::test_amg(solver, presmoothing, postsmoothing, cycle, smoother, filename));
}

INSTANTIATE_TEST_CASE_P(amg, 
                        amg_parameters,
                        testing::Combine(testing::ValuesIn(amg_solvers), 
                                         testing::ValuesIn(amg_presmoothing_iterations),
                                         testing::ValuesIn(amg_postsmoothing_iterations),
                                         testing::ValuesIn(cycle_types), 
                                         testing::ValuesIn(smoother_type),
                                         testing::ValuesIn(amg_matrix_files)),
                        [](const testing::TestParamInfo<amg_parameters::ParamType>& info) 
                        {
                          std::string solver = Testing::AMGSolverToString(std::get<0>(info.param));
                          std::string presmoothing_iterations = std::to_string(std::get<1>(info.param));
                          std::string postsmoothing_iterations = std::to_string(std::get<2>(info.param));
                          std::string cycle = Testing::CycleToString(std::get<3>(info.param));
                          std::string smoother = Testing::SmootherToString(std::get<4>(info.param));
                          std::string filename = std::get<5>(info.param);
                          for(size_t i = 0; i < filename.length(); i++)
                          {
                            if(filename[i] == '.')
                            {
                              filename[i] = '_';
                            }
                          }
                          return solver + "_" + presmoothing_iterations + "_" + postsmoothing_iterations + "_" + cycle + "_" + smoother + "_" + filename;
                        }
                        );

#endif