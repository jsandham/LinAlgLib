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
#include <queue>
#include <stack>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "test_arguments.h"
#include "test_enums.h"

namespace YAML
{
    template <>
    struct convert<Testing::preconditioner>
    {
        static Node encode(const Testing::preconditioner& rhs)
        {
            Node node;
            switch(rhs)
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

        static bool decode(const Node& node, Testing::preconditioner& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "Jacobi")
            {
                rhs = Testing::preconditioner::Jacobi;
            }
            if(type == "GaussSeidel")
            {
                rhs = Testing::preconditioner::GaussSeidel;
            }
            if(type == "SOR")
            {
                rhs = Testing::preconditioner::SOR;
            }
            if(type == "SymmGaussSeidel")
            {
                rhs = Testing::preconditioner::SymmGaussSeidel;
            }
            else if(type == "IC")
            {
                rhs = Testing::preconditioner::IC;
            }
            else if(type == "ILU")
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

    template <>
    struct convert<linalg::Cycle>
    {
        static Node encode(const linalg::Cycle& rhs)
        {
            Node node;
            switch(rhs)
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

        static bool decode(const Node& node, linalg::Cycle& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "Fcycle")
            {
                rhs = linalg::Cycle::Fcycle;
            }
            else if(type == "Vcycle")
            {
                rhs = linalg::Cycle::Vcycle;
            }
            else if(type == "Wcycle")
            {
                rhs = linalg::Cycle::Wcycle;
            }

            return true;
        }
    };

    template <>
    struct convert<linalg::Smoother>
    {
        static Node encode(const linalg::Smoother& rhs)
        {
            Node node;
            switch(rhs)
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

        static bool decode(const Node& node, linalg::Smoother& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "Jacobi")
            {
                rhs = linalg::Smoother::Jacobi;
            }
            else if(type == "Gauss_Seidel")
            {
                rhs = linalg::Smoother::Gauss_Seidel;
            }
            else if(type == "Symm_Gauss_Seidel")
            {
                rhs = linalg::Smoother::Symm_Gauss_Seidel;
            }
            else if(type == "SOR")
            {
                rhs = linalg::Smoother::SOR;
            }
            else if(type == "SSOR")
            {
                rhs = linalg::Smoother::SSOR;
            }

            return true;
        }
    };
}

template <typename T>
std::vector<T> read_values(const std::string& group,
                           const std::string& label,
                           const YAML::Node&  node,
                           T                  default_value)
{
    std::vector<T> values;

    if(node[group][label].IsDefined())
    {
        if(node[group][label].IsSequence())
        {
            values = node[group][label].as<std::vector<T>>();
        }
        else
        {
            values.push_back(node[group][label].as<T>());
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

inline Testing::Category StringToCategory(const std::string& str)
{
    // Static map for efficiency. It's initialized only once.
    static const std::unordered_map<std::string, Testing::Category> categoryMap
        = {{"IterativeSolvers", Testing::Category::IterativeSolvers},
           {"Math", Testing::Category::Math},
           {"Primitive", Testing::Category::Primitive}};

    // Find the string in the map
    auto it = categoryMap.find(str);

    // Return the corresponding enum value or a default value
    if(it != categoryMap.end())
    {
        return it->second;
    }

    return Testing::Category::Unknown;
}

inline Testing::Fixture StringToFixture(const std::string& str)
{
    // Static map for efficiency. It's initialized only once.
    static const std::unordered_map<std::string, Testing::Fixture> fixtureMap
        = {{"Jacobi", Testing::Fixture::Jacobi},
           {"GaussSeidel", Testing::Fixture::GaussSeidel},
           {"SOR", Testing::Fixture::SOR},
           {"SymmGaussSeidel", Testing::Fixture::SymmGaussSeidel},
           {"SSOR", Testing::Fixture::SSOR},
           {"CG", Testing::Fixture::CG},
           {"BICGSTAB", Testing::Fixture::BICGSTAB},
           {"GMRES", Testing::Fixture::GMRES},
           {"UAAMG", Testing::Fixture::UAAMG},
           {"SAAMG", Testing::Fixture::SAAMG},
           {"RSAMG", Testing::Fixture::RSAMG},
           {"SpMV", Testing::Fixture::SpMV},
           {"SpGEMM", Testing::Fixture::SpGEMM},
           {"SpGEAM", Testing::Fixture::SpGEAM},
           {"Transpose", Testing::Fixture::Transpose},
           {"ExclusiveScan", Testing::Fixture::ExclusiveScan}};

    // Find the string in the map
    auto it = fixtureMap.find(str);

    // Return the corresponding enum value or a default value
    if(it != fixtureMap.end())
    {
        return it->second;
    }

    return Testing::Fixture::Unknown;
}

// Helper struct to hold all parameter vectors
struct TestParameters
{
    std::vector<std::string>             matrices;
    std::vector<Testing::preconditioner> preconds;
    std::vector<linalg::Cycle>           cycles;
    std::vector<linalg::Smoother>        smoothers;
    std::vector<int>                     presmoothings;
    std::vector<int>                     postsmoothings;
    std::vector<int>                     max_iters;
    std::vector<double>                  tols;
    std::vector<double>                  omegas;
};

inline std::vector<Testing::Arguments> generate_tests(const std::string category,
                                                      const std::string fixture,
                                                      const std::string filepath)
{
    YAML::Node                      root_node = YAML::LoadFile(correct_test_filepath(filepath));
    std::vector<Testing::Arguments> tests;

    std::cout << "category: " << category << " fixture: " << fixture << " filepath: " << filepath
              << std::endl;

    for(YAML::const_iterator it = root_node["Tests"].begin(); it != root_node["Tests"].end(); ++it)
    {
        std::string group = it->first.as<std::string>();

        TestParameters params;
        params.matrices = read_values<std::string>(group, "matrix_file", root_node["Tests"], "");
        params.preconds = read_values<Testing::preconditioner>(
            group, "precond", root_node["Tests"], Testing::preconditioner::None);
        params.cycles
            = read_values<linalg::Cycle>(group, "cycle", root_node["Tests"], linalg::Cycle::Vcycle);
        params.smoothers = read_values<linalg::Smoother>(
            group, "smoother", root_node["Tests"], linalg::Smoother::Jacobi);
        params.presmoothings  = read_values<int>(group, "presmoothing", root_node["Tests"], 1);
        params.postsmoothings = read_values<int>(group, "postsmoothing", root_node["Tests"], 1);
        params.max_iters      = read_values<int>(group, "max_iters", root_node["Tests"], 1000);
        params.tols           = read_values<double>(group, "tol", root_node["Tests"], 1e-06);
        params.omegas         = read_values<double>(group, "omega", root_node["Tests"], 0.66667);

        size_t total_tests = params.matrices.size() * params.preconds.size() * params.cycles.size()
                             * params.smoothers.size() * params.presmoothings.size()
                             * params.postsmoothings.size() * params.max_iters.size()
                             * params.tols.size() * params.omegas.size();
        std::cout << "total_tests: " << total_tests << std::endl;

        // Reserve memory once to avoid multiple reallocations
        tests.reserve(tests.size() + total_tests);

        // A tuple for each item on the stack: {current_args, depth}
        using StackItem = std::tuple<Testing::Arguments, size_t>;
        std::stack<StackItem> s;

        // Push the initial state onto the stack
        Testing::Arguments initial_args;
        initial_args.category = StringToCategory(category);
        initial_args.fixture  = StringToFixture(fixture);
        initial_args.group    = group;
        s.push({initial_args, 0});

        std::cout << "initial_args.category: " << (int)initial_args.category
                  << " initial_args.fixture: " << (int)initial_args.fixture << std::endl;

        // The main loop
        while(!s.empty())
        {
            StackItem current_item = s.top();
            s.pop();

            Testing::Arguments current_args = std::get<0>(current_item);
            size_t             depth        = std::get<1>(current_item);

            // Base Case: A full combination is found
            if(depth == 9 /*param_lists.size()*/)
            {
                tests.push_back(current_args);
                continue;
            }

            // Recursive Step (Iterative): Expand to the next level
            // This part is the tricky bit due to different types
            if(depth == 0)
            { // Matrix Files
                for(const auto& matrix_file : params.matrices)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.filename           = matrix_file;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 1)
            {
                for(const auto& precond : params.preconds)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.precond            = precond;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 2)
            {
                for(const auto& cycle : params.cycles)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.cycle              = cycle;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 3)
            {
                for(const auto& smoother : params.smoothers)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.smoother           = smoother;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 4)
            {
                for(const auto& presmoothing : params.presmoothings)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.presmoothing       = presmoothing;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 5)
            {
                for(const auto& postsmoothing : params.postsmoothings)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.postsmoothing      = postsmoothing;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 6)
            {
                for(const auto& max_iters : params.max_iters)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.max_iters          = max_iters;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 7)
            {
                for(const auto& tol : params.tols)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.tol                = tol;
                    s.push({next_args, depth + 1});
                }
            }
            else if(depth == 8)
            {
                for(const auto& omega : params.omegas)
                {
                    Testing::Arguments next_args = current_args;
                    next_args.omega              = omega;
                    s.push({next_args, depth + 1});
                }
            }
        }
    }

    return tests;
}

#endif