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
    struct convert<testing::backend>
    {
        static Node encode(const testing::backend& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::backend::CPU:
                node = "CPU";
                break;
            case testing::backend::GPU:
                node = "GPU";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::backend& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "CPU")
            {
                rhs = testing::backend::CPU;
            }
            if(type == "GPU")
            {
                rhs = testing::backend::GPU;
            }

            return true;
        }
    };

    template <>
    struct convert<testing::uplo>
    {
        static Node encode(const testing::uplo& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::uplo::lower:
                node = "lower";
                break;
            case testing::uplo::upper:
                node = "upper";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::uplo& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "lower")
            {
                rhs = testing::uplo::lower;
            }
            else if(type == "upper")
            {
                rhs = testing::uplo::upper;
            }

            return true;
        }
    };

    template <>
    struct convert<testing::preconditioner>
    {
        static Node encode(const testing::preconditioner& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::preconditioner::jacobi:
                node = "jacobi";
                break;
            case testing::preconditioner::gauss_seidel:
                node = "gauss_seidel";
                break;
            case testing::preconditioner::SOR:
                node = "SOR";
                break;
            case testing::preconditioner::symmetric_gauss_seidel:
                node = "symmetric_gauss_seidel";
                break;
            case testing::preconditioner::IC:
                node = "IC";
                break;
            case testing::preconditioner::ILU:
                node = "ILU";
                break;
            case testing::preconditioner::none:
                node = "none";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::preconditioner& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "jacobi")
            {
                rhs = testing::preconditioner::jacobi;
            }
            if(type == "gauss_seidel")
            {
                rhs = testing::preconditioner::gauss_seidel;
            }
            if(type == "SOR")
            {
                rhs = testing::preconditioner::SOR;
            }
            if(type == "symmetric_gauss_seidel")
            {
                rhs = testing::preconditioner::symmetric_gauss_seidel;
            }
            if(type == "SSOR")
            {
                rhs = testing::preconditioner::SSOR;
            }
            else if(type == "IC")
            {
                rhs = testing::preconditioner::IC;
            }
            else if(type == "ILU")
            {
                rhs = testing::preconditioner::ILU;
            }
            else if(type == "none")
            {
                rhs = testing::preconditioner::none;
            }

            return true;
        }
    };

    template <>
    struct convert<testing::cycle_type>
    {
        static Node encode(const testing::cycle_type& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::cycle_type::fcycle:
                node = "fcycle";
                break;
            case testing::cycle_type::vcycle:
                node = "vcycle";
                break;
            case testing::cycle_type::wcycle:
                node = "wcycle";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::cycle_type& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "fcycle")
            {
                rhs = testing::cycle_type::fcycle;
            }
            else if(type == "vcycle")
            {
                rhs = testing::cycle_type::vcycle;
            }
            else if(type == "wcycle")
            {
                rhs = testing::cycle_type::wcycle;
            }

            return true;
        }
    };

    template <>
    struct convert<testing::smoother_type>
    {
        static Node encode(const testing::smoother_type& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::smoother_type::jacobi:
                node = "jacobi";
                break;
            case testing::smoother_type::gauss_seidel:
                node = "gauss_seidel";
                break;
            case testing::smoother_type::symmetric_gauss_seidel:
                node = "symmetric_gauss_seidel";
                break;
            case testing::smoother_type::SOR:
                node = "SOR";
                break;
            case testing::smoother_type::SSOR:
                node = "SSOR";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::smoother_type& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "jacobi")
            {
                rhs = testing::smoother_type::jacobi;
            }
            else if(type == "gauss_seidel")
            {
                rhs = testing::smoother_type::gauss_seidel;
            }
            else if(type == "symmetric_gauss_seidel")
            {
                rhs = testing::smoother_type::symmetric_gauss_seidel;
            }
            else if(type == "SOR")
            {
                rhs = testing::smoother_type::SOR;
            }
            else if(type == "SSOR")
            {
                rhs = testing::smoother_type::SSOR;
            }

            return true;
        }
    };

    template <>
    struct convert<testing::pivoting_strategy>
    {
        static Node encode(const testing::pivoting_strategy& rhs)
        {
            Node node;
            switch(rhs)
            {
            case testing::pivoting_strategy::partial:
                node = "partial";
                break;
            case testing::pivoting_strategy::none:
                node = "none";
                break;
            }

            return node;
        }

        static bool decode(const Node& node, testing::pivoting_strategy& rhs)
        {
            std::string type = node.as<std::string>();
            if(type == "partial")
            {
                rhs = testing::pivoting_strategy::partial;
            }
            else if(type == "none")
            {
                rhs = testing::pivoting_strategy::none;
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

inline testing::category string_to_category(const std::string& str)
{
    // Static map for efficiency. It's initialized only once.
    static const std::unordered_map<std::string, testing::category> category_map
        = {{"iterative_solvers", testing::category::iterative_solvers},
           {"direct_solvers", testing::category::direct_solvers},
           {"math", testing::category::math},
           {"primitive", testing::category::primitive},
           {"csr_matrix", testing::category::csr_matrix}};

    // Find the string in the map
    auto it = category_map.find(str);

    // Return the corresponding enum value or a default value
    if(it != category_map.end())
    {
        return it->second;
    }

    return testing::category::unknown;
}

inline testing::fixture string_to_fixture(const std::string& str)
{
    // Static map for efficiency. It's initialized only once.
    static const std::unordered_map<std::string, testing::fixture> fixture_map
        = {{"jacobi", testing::fixture::jacobi},
           {"gauss_seidel", testing::fixture::gauss_seidel},
           {"SOR", testing::fixture::SOR},
           {"symmetric_gauss_seidel", testing::fixture::symmetric_gauss_seidel},
           {"SSOR", testing::fixture::SSOR},
           {"CG", testing::fixture::CG},
           {"BICGSTAB", testing::fixture::BICGSTAB},
           {"GMRES", testing::fixture::GMRES},
           {"UAAMG", testing::fixture::UAAMG},
           {"SAAMG", testing::fixture::SAAMG},
           {"RSAMG", testing::fixture::RSAMG},
           {"multiply_by_vector", testing::fixture::multiply_by_vector},
           {"multiply_by_matrix", testing::fixture::multiply_by_matrix},
           {"triangular_solve", testing::fixture::triangular_solve},
           {"compute_incomplete_cholesky_factorization",
            testing::fixture::compute_incomplete_cholesky_factorization},
           {"compute_incomplete_cholesky_factorization_dense",
            testing::fixture::compute_incomplete_cholesky_factorization_dense},
           {"SpTRSV", testing::fixture::SpTRSV},
           {"SpGEAM", testing::fixture::SpGEAM},
           {"transpose", testing::fixture::transpose},
           {"transpose_dense", testing::fixture::transpose_dense},
           {"CSRIC0", testing::fixture::CSRIC0},
           {"CSRILU0", testing::fixture::CSRILU0},
           {"tridiagonal_solver", testing::fixture::tridiagonal_solver},
           {"ruiz_scaling", testing::fixture::ruiz_scaling},
           {"symmetric_ruiz_scaling", testing::fixture::symmetric_ruiz_scaling},
           {"exclusive_scan", testing::fixture::exclusive_scan}};

    // Find the string in the map
    auto it = fixture_map.find(str);

    // Return the corresponding enum value or a default value
    if(it != fixture_map.end())
    {
        return it->second;
    }

    return testing::fixture::unknown;
}

// Helper struct to hold all parameter vectors
struct TestParameters
{
    std::vector<std::string>                matrices;
    std::vector<testing::backend>           backends;
    std::vector<testing::uplo>              uplos;
    std::vector<testing::preconditioner>    precond_types;
    std::vector<testing::cycle_type>        cycle_types;
    std::vector<testing::smoother_type>     smoother_types;
    std::vector<testing::pivoting_strategy> pivoting_strategies;
    std::vector<int>                        presmoothings;
    std::vector<int>                        postsmoothings;
    std::vector<int>                        max_iters;
    std::vector<int>                        m_values;
    std::vector<int>                        n_values;
    std::vector<double>                     tols;
    std::vector<double>                     omegas;
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

inline std::vector<testing::Arguments> generate_tests(const std::string category,
                                                      const std::string fixture,
                                                      const std::string filepath)
{
    const testing::category category_enum = string_to_category(category);
    const testing::fixture  fixture_enum  = string_to_fixture(fixture);

    const std::string resolved_filepath = correct_test_filepath(filepath);
    YAML::Node        root_node         = YAML::LoadFile(resolved_filepath);
    const YAML::Node  tests_node        = root_node["Tests"];

    std::vector<testing::Arguments> tests;

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
        params.backends       = read_group_values("backend", testing::backend::CPU);
        params.uplos          = read_group_values("uplo", testing::uplo::lower);
        params.precond_types  = read_group_values("precond", testing::preconditioner::none);
        params.cycle_types    = read_group_values("cycle", testing::cycle_type::none);
        params.smoother_types = read_group_values("smoother", testing::smoother_type::none);
        params.pivoting_strategies
            = read_group_values("pivoting_strategy", testing::pivoting_strategy::none);
        params.presmoothings  = read_group_values("presmoothing", -1);
        params.postsmoothings = read_group_values("postsmoothing", -1);
        params.max_iters      = read_group_values("max_iters", -1);
        params.m_values       = read_group_values("m", -1);
        params.n_values       = read_group_values("n", -1);
        params.tols           = read_group_values("tol", -1.0);
        params.omegas         = read_group_values("omega", -1.0);

        size_t total_tests = 1;
        total_tests *= params.matrices.size();
        total_tests *= params.backends.size();
        total_tests *= params.uplos.size();
        total_tests *= params.precond_types.size();
        total_tests *= params.cycle_types.size();
        total_tests *= params.smoother_types.size();
        total_tests *= params.pivoting_strategies.size();
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
            [&](const std::string&         filename,
                testing::backend           backend,
                testing::uplo              uplo,
                testing::preconditioner    precond,
                testing::cycle_type        cycle_type,
                testing::smoother_type     smoother_type,
                testing::pivoting_strategy pivoting_strategy,
                int                        presmoothing,
                int                        postsmoothing,
                int                        max_iters,
                int                        m,
                int                        n,
                double                     tol,
                double                     omega) {
                tests.emplace_back(testing::Arguments{
                    category_enum,
                    fixture_enum,
                    group,
                    filename,
                    backend,
                    uplo,
                    precond,
                    cycle_type,
                    smoother_type,
                    pivoting_strategy,
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
            params.backends,
            params.uplos,
            params.precond_types,
            params.cycle_types,
            params.smoother_types,
            params.pivoting_strategies,
            params.presmoothings,
            params.postsmoothings,
            params.max_iters,
            params.m_values,
            params.n_values,
            params.tols,
            params.omegas);

        // for(size_t i = 0; i < total_tests; i++)
        // {
        //     std::cout << "Generated test name: " << tests[i].generate_test_name() << std::endl;
        // }
    }

    return tests;
}

#endif
