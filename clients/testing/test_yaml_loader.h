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
            if(type == "SSOR")
            {
                rhs = Testing::preconditioner::SSOR;
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
    struct convert<Testing::cycle_type>
    {
        static Node encode(const Testing::cycle_type& rhs)
        {
            Node node;
            switch(rhs)
            {
            case Testing::cycle_type::Fcycle:
                node = "Fcycle";
                break;
            case Testing::cycle_type::Vcycle:
                node = "Vcycle";
                break;
            case Testing::cycle_type::Wcycle:
                node = "Wcycle";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, Testing::cycle_type& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "Fcycle")
            {
                rhs = Testing::cycle_type::Fcycle;
            }
            else if(type == "Vcycle")
            {
                rhs = Testing::cycle_type::Vcycle;
            }
            else if(type == "Wcycle")
            {
                rhs = Testing::cycle_type::Wcycle;
            }

            return true;
        }
    };

    template <>
    struct convert<Testing::smoother_type>
    {
        static Node encode(const Testing::smoother_type& rhs)
        {
            Node node;
            switch(rhs)
            {
            case Testing::smoother_type::Jacobi:
                node = "Jacobi";
                break;
            case Testing::smoother_type::Gauss_Seidel:
                node = "Gauss_Seidel";
                break;
            case Testing::smoother_type::Symm_Gauss_Seidel:
                node = "Symm_Gauss_Seidel";
                break;
            case Testing::smoother_type::SOR:
                node = "SOR";
                break;
            case Testing::smoother_type::SSOR:
                node = "SSOR";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, Testing::smoother_type& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "Jacobi")
            {
                rhs = Testing::smoother_type::Jacobi;
            }
            else if(type == "Gauss_Seidel")
            {
                rhs = Testing::smoother_type::Gauss_Seidel;
            }
            else if(type == "Symm_Gauss_Seidel")
            {
                rhs = Testing::smoother_type::Symm_Gauss_Seidel;
            }
            else if(type == "SOR")
            {
                rhs = Testing::smoother_type::SOR;
            }
            else if(type == "SSOR")
            {
                rhs = Testing::smoother_type::SSOR;
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
           {"SpTRSV", Testing::Fixture::SpTRSV},
           {"SpGEMM", Testing::Fixture::SpGEMM},
           {"SpGEAM", Testing::Fixture::SpGEAM},
           {"Transpose", Testing::Fixture::Transpose},
           {"CSRIC0", Testing::Fixture::CSRIC0},
           {"CSRILU0", Testing::Fixture::CSRILU0},
           {"TridiagonalSolver", Testing::Fixture::TridiagonalSolver},
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
    std::vector<Testing::preconditioner> precond_types;
    std::vector<Testing::cycle_type>     cycle_types;
    std::vector<Testing::smoother_type>  smoother_types;
    std::vector<int>                     presmoothings;
    std::vector<int>                     postsmoothings;
    std::vector<int>                     max_iters;
    std::vector<int>                     m_values;
    std::vector<int>                     n_values;
    std::vector<double>                  tols;
    std::vector<double>                  omegas;
};

template <std::size_t I = 0, typename F, typename Tuple, typename... Args>
void for_each_combination_impl(F&& f, const Tuple& containers, Args&&... args)
{
    if constexpr(I == std::tuple_size_v<std::remove_reference_t<Tuple>>)
    {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
    else
    {
        for(const auto& item : std::get<I>(containers))
        {
            for_each_combination_impl<I + 1>(
                std::forward<F>(f), containers, std::forward<Args>(args)..., item);
        }
    }
}

template <typename F, typename... Containers>
void for_each_combination(F&& f, const Containers&... containers)
{
    auto tupled = std::forward_as_tuple(containers...);
    for_each_combination_impl(std::forward<F>(f), tupled);
}

inline std::vector<Testing::Arguments> generate_tests(const std::string category,
                                                      const std::string fixture,
                                                      const std::string filepath)
{
    const Testing::Category category_enum = StringToCategory(category);
    const Testing::Fixture  fixture_enum  = StringToFixture(fixture);

    const std::string resolved_filepath = correct_test_filepath(filepath);
    YAML::Node        root_node         = YAML::LoadFile(resolved_filepath);
    const YAML::Node  tests_node        = root_node["Tests"];

    std::vector<Testing::Arguments> tests;

    std::cout << "category: " << category << " fixture: " << fixture << " filepath: " << filepath
              << std::endl;

    if(!tests_node || !tests_node.IsMap())
    {
        return tests;
    }

    for(const auto& test_entry : tests_node)
    {
        std::string group = test_entry.first.as<std::string>();

        std::cout << "group: " << group << std::endl;

        auto read_group_values = [&](const std::string& label, auto default_value) {
            using value_type = decltype(default_value);
            return read_values<value_type>(group, label, tests_node, default_value);
        };

        TestParameters params;
        params.matrices       = read_group_values("matrix_file", std::string(""));
        params.precond_types  = read_group_values("precond", Testing::preconditioner::None);
        params.cycle_types    = read_group_values("cycle", Testing::cycle_type::None);
        params.smoother_types = read_group_values("smoother", Testing::smoother_type::None);
        params.presmoothings  = read_group_values("presmoothing", -1);
        params.postsmoothings = read_group_values("postsmoothing", -1);
        params.max_iters      = read_group_values("max_iters", -1);
        params.m_values       = read_group_values("m", -1);
        params.n_values       = read_group_values("n", -1);
        params.tols           = read_group_values("tol", -1.0);
        params.omegas         = read_group_values("omega", -1.0);

        size_t total_tests = 1;
        total_tests *= params.matrices.size();
        total_tests *= params.precond_types.size();
        total_tests *= params.cycle_types.size();
        total_tests *= params.smoother_types.size();
        total_tests *= params.presmoothings.size();
        total_tests *= params.postsmoothings.size();
        total_tests *= params.max_iters.size();
        total_tests *= params.m_values.size();
        total_tests *= params.n_values.size();
        total_tests *= params.tols.size();
        total_tests *= params.omegas.size();

        std::cout << "total_tests: " << total_tests << std::endl;

        // Reserve once per group to avoid repeated reallocations while appending.
        tests.reserve(tests.size() + total_tests);

        for_each_combination(
            [&](const std::string&      filename,
                Testing::preconditioner precond,
                Testing::cycle_type     cycle_type,
                Testing::smoother_type  smoother_type,
                int                     presmoothing,
                int                     postsmoothing,
                int                     max_iters,
                int                     m,
                int                     n,
                double                  tol,
                double                  omega) {
                tests.emplace_back(Testing::Arguments{
                    category_enum,
                    fixture_enum,
                    group,
                    filename,
                    precond,
                    cycle_type,
                    smoother_type,
                    presmoothing,
                    postsmoothing,
                    max_iters,
                    m,
                    n,
                    tol,
                    omega,
                });
            },
            params.matrices,
            params.precond_types,
            params.cycle_types,
            params.smoother_types,
            params.presmoothings,
            params.postsmoothings,
            params.max_iters,
            params.m_values,
            params.n_values,
            params.tols,
            params.omegas);

        for(size_t i = 0; i < total_tests; i++)
        {
            std::cout << "Generated test name: " << tests[i].generate_test_name() << std::endl;
        }
    }

    return tests;
}

#endif
