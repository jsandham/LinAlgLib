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

#ifndef TEST_YAML_LOADER_H__
#define TEST_YAML_LOADER_H__

#include <iostream>
#include <vector>
#include <string>
#include <queue>

#include <yaml-cpp/yaml.h>

#include "test_arguments.h"
#include "test_enums.h"

namespace YAML
{
    template <> struct convert<Testing::Solver>
    {
        static Node encode(const Testing::Solver &rhs)
        {
            Node node;
            switch (rhs)
            {
            case Testing::Solver::Jacobi:
                node = "Jacobi";
                break;
            case Testing::Solver::GaussSeidel:
                node = "GaussSeidel";
                break;
            case Testing::Solver::SymmGaussSeidel:
                node = "SymmGaussSeidel";
                break;
            case Testing::Solver::SOR:
                node = "SOR";
                break;
            case Testing::Solver::SSOR:
                node = "SSOR";
                break;
            case Testing::Solver::CG:
                node = "CG";
                break;
            case Testing::Solver::GMRES:
                node = "GMRES";
                break;
            case Testing::Solver::BICGSTAB:
                node = "BICGSTAB";
                break;
            case Testing::Solver::UAAMG:
                node = "UAAMG";
                break;
            case Testing::Solver::SAAMG:
                node = "SAAMG";
                break;
            case Testing::Solver::RSAMG:
                node = "RSAMG";
                break;
            }
    
            return node;
        }
    
        static bool decode(const Node &node, Testing::Solver &rhs)
        {
            std::string type = node.as<std::string>();
            if (type == "Jacobi")
            {
                rhs = Testing::Solver::Jacobi;
            }
            else if (type == "GaussSeidel")
            {
                rhs = Testing::Solver::GaussSeidel;
            }
            else if (type == "SymmGaussSeidel")
            {
                rhs = Testing::Solver::SymmGaussSeidel;
            }
            else if (type == "SOR")
            {
                rhs = Testing::Solver::SOR;
            }
            else if (type == "SSOR")
            {
                rhs = Testing::Solver::SSOR;
            }
            else if (type == "CG")
            {
                rhs = Testing::Solver::CG;
            }
            else if (type == "GMRES")
            {
                rhs = Testing::Solver::GMRES;
            }
            else if (type == "BICGSTAB")
            {
                rhs = Testing::Solver::BICGSTAB;
            }
            else if (type == "UAAMG")
            {
                rhs = Testing::Solver::UAAMG;
            }
            else if (type == "SAAMG")
            {
                rhs = Testing::Solver::SAAMG;
            }
            else if (type == "RSAMG")
            {
                rhs = Testing::Solver::RSAMG;
            }
            

            return true;
        }
    };

template <> struct convert<Testing::ClassicalSolver>
{
    static Node encode(const Testing::ClassicalSolver &rhs)
    {
        Node node;
        switch (rhs)
        {
        case Testing::ClassicalSolver::Jacobi:
            node = "Jacobi";
            break;
        case Testing::ClassicalSolver::GaussSeidel:
            node = "GaussSeidel";
            break;
        case Testing::ClassicalSolver::SymmGaussSeidel:
            node = "SymmGaussSeidel";
            break;
        case Testing::ClassicalSolver::SOR:
            node = "SOR";
            break;
        case Testing::ClassicalSolver::SSOR:
            node = "SSOR";
            break;
        }

        return node;
    }

    static bool decode(const Node &node, Testing::ClassicalSolver &rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Jacobi")
        {
            rhs = Testing::ClassicalSolver::Jacobi;
        }
        else if (type == "GaussSeidel")
        {
            rhs = Testing::ClassicalSolver::GaussSeidel;
        }
        else if (type == "SymmGaussSeidel")
        {
            rhs = Testing::ClassicalSolver::SymmGaussSeidel;
        }
        else if (type == "SOR")
        {
            rhs = Testing::ClassicalSolver::SOR;
        }
        else if (type == "SSOR")
        {
            rhs = Testing::ClassicalSolver::SSOR;
        }

        return true;
    }
};

template <> struct convert<Testing::preconditioner>
{
    static Node encode(const Testing::preconditioner &rhs)
    {
        Node node;
        switch (rhs)
        {
        case Testing::preconditioner::Jacobi:
            node = "Jacobi";
            break;
        case Testing::preconditioner::GaussSeidel:
            node = "GaussSeidel";
            break;
        case Testing::preconditioner::SOR:
            node = "SOR";
            break;
        case Testing::preconditioner::SymmGaussSeidel:
            node = "SymmGaussSeidel";
            break;
        case Testing::preconditioner::IC:
            node = "IC";
            break;
        case Testing::preconditioner::ILU:
            node = "ILU";
            break;
        case Testing::preconditioner::None:
            node = "None";
            break;
        }

        return node;
    }

    static bool decode(const Node &node, Testing::preconditioner &rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Jacobi")
        {
            rhs = Testing::preconditioner::Jacobi;
        }
        if (type == "GaussSeidel")
        {
            rhs = Testing::preconditioner::GaussSeidel;
        }
        if (type == "SOR")
        {
            rhs = Testing::preconditioner::SOR;
        }
        if (type == "SymmGaussSeidel")
        {
            rhs = Testing::preconditioner::SymmGaussSeidel;
        }
        else if (type == "IC")
        {
            rhs = Testing::preconditioner::IC;
        }
        else if (type == "ILU")
        {
            rhs = Testing::preconditioner::ILU;
        }
        else if(type == "None")
        {
            rhs = Testing::preconditioner::None;
        }

        return true;
    }
};

template <> struct convert<linalg::Cycle>
{
    static Node encode(const linalg::Cycle &rhs)
    {
        Node node;
        switch (rhs)
        {
        case linalg::Cycle::Fcycle:
            node = "Fcycle";
            break;
        case linalg::Cycle::Vcycle:
            node = "Vcycle";
            break;
        case linalg::Cycle::Wcycle:
            node = "Wcycle";
            break;
        }

        return node;
    }

    static bool decode(const Node &node, linalg::Cycle &rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Fcycle")
        {
            rhs = linalg::Cycle::Fcycle;
        }
        else if (type == "Vcycle")
        {
            rhs = linalg::Cycle::Vcycle;
        }
        else if (type == "Wcycle")
        {
            rhs = linalg::Cycle::Wcycle;
        }

        return true;
    }
};

template <> struct convert<linalg::Smoother>
{
    static Node encode(const linalg::Smoother &rhs)
    {
        Node node;
        switch (rhs)
        {
        case linalg::Smoother::Jacobi:
            node = "Jacobi";
            break;
        case linalg::Smoother::Gauss_Seidel:
            node = "Gauss_Seidel";
            break;
        case linalg::Smoother::Symm_Gauss_Seidel:
            node = "Symm_Gauss_Seidel";
            break;
        case linalg::Smoother::SOR:
            node = "SOR";
            break;
        case linalg::Smoother::SSOR:
            node = "SSOR";
            break;
        }

        return node;
    }

    static bool decode(const Node &node, linalg::Smoother &rhs)
    {
        std::string type = node.as<std::string>();
        if (type == "Jacobi")
        {
            rhs = linalg::Smoother::Jacobi;
        }
        else if (type == "Gauss_Seidel")
        {
            rhs = linalg::Smoother::Gauss_Seidel;
        }
        else if (type == "Symm_Gauss_Seidel")
        {
            rhs = linalg::Smoother::Symm_Gauss_Seidel;
        }
        else if (type == "SOR")
        {
            rhs = linalg::Smoother::SOR;
        }
        else if (type == "SSOR")
        {
            rhs = linalg::Smoother::SSOR;
        }

        return true;
    }
};
}

template<typename T>
std::vector<T> read_values(const std::string& category, const std::string& label, const YAML::Node& node, T default_value)
{
    std::vector<T> values;

    if(node[category][label].IsDefined())
    {
        if(node[category][label].IsSequence())
        {
            values = node[category][label].as<std::vector<T>>();
        }
        else
        {
            values.push_back(node[category][label].as<T>());
        }

        return values;
    }

    values.push_back(default_value);

    return values;
}

inline std::string correct_test_filepath(const std::string& filepath)
{
#if defined(_WIN32) || defined(WIN32)
    return "../" + filepath;
#else
    return filepath;
#endif
}

inline std::vector<Testing::Arguments> generate_tests(const std::string filepath)
{
    std::cout << "filepath: " << correct_test_filepath(filepath) << std::endl;
    YAML::Node root_node = YAML::LoadFile(correct_test_filepath(filepath));

    size_t index = 0;
    std::vector<Testing::Arguments> tests;

    for (YAML::const_iterator it = root_node["Tests"].begin(); it != root_node["Tests"].end(); ++it)
    {
        std::string category = it->first.as<std::string>();

        std::cout << "category: " << category << std::endl;

        std::vector<Testing::Solver> solvers = read_values<Testing::Solver>(category, "solver", root_node["Tests"], Testing::Solver::Jacobi);
        std::vector<std::string> matrices = read_values<std::string>(category, "matrix_file", root_node["Tests"], "");
        std::vector<Testing::preconditioner> preconds = read_values<Testing::preconditioner>(category, "precond", root_node["Tests"], Testing::preconditioner::None);
        std::vector<linalg::Cycle> cycles = read_values<linalg::Cycle>(category, "cycle", root_node["Tests"], linalg::Cycle::Vcycle);
        std::vector<linalg::Smoother> smoothers = read_values<linalg::Smoother>(category, "smoother", root_node["Tests"], linalg::Smoother::Jacobi);
        std::vector<int> presmoothings = read_values<int>(category, "presmoothing", root_node["Tests"], 1);
        std::vector<int> postsmoothings = read_values<int>(category, "postsmoothing", root_node["Tests"], 1);
        std::vector<int> max_iters = read_values<int>(category, "max_iters", root_node["Tests"], 1000);
        std::vector<double> tols = read_values<double>(category, "tol", root_node["Tests"], 1e-06);
        std::vector<double> omegas = read_values<double>(category, "omega", root_node["Tests"], 0.66667);

        size_t total_tests = solvers.size() * matrices.size() * preconds.size() * cycles.size() * smoothers.size() * presmoothings.size() *
                            postsmoothings.size() * max_iters.size() * tols.size() * omegas.size();

        std::cout << "total_tests: " << total_tests << std::endl;

        tests.resize(tests.size() + total_tests);

        for(size_t i = 0; i < solvers.size(); i++)
        {
            for(size_t j = 0; j < matrices.size(); j++)
            {
                for(size_t k = 0; k < preconds.size(); k++)
                {
                    for(size_t l = 0; l < cycles.size(); l++)
                    {
                        for(size_t m = 0; m < smoothers.size(); m++)
                        {
                            for(size_t n = 0; n < presmoothings.size(); n++)
                            {
                                for(size_t o = 0; o < postsmoothings.size(); o++)
                                {
                                    for(size_t p = 0; p < max_iters.size(); p++)
                                    {
                                        for(size_t q = 0; q < tols.size(); q++)
                                        {
                                            for(size_t r = 0; r < omegas.size(); r++)
                                            {
                                                tests[index].category = category;
                                                tests[index].solver = solvers[i]; 
                                                tests[index].filename = matrices[j]; 
                                                tests[index].precond = preconds[k]; 
                                                tests[index].cycle = cycles[l]; 
                                                tests[index].smoother = smoothers[m]; 
                                                tests[index].presmoothing = presmoothings[n]; 
                                                tests[index].postsmoothing = postsmoothings[o]; 
                                                tests[index].max_iters = max_iters[p]; 
                                                tests[index].tol = tols[q]; 
                                                tests[index].omega = omegas[r];
                                                index++;                    
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return tests;
}

#endif