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

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "test_arguments.h"
#include "test_functions.h"
#include "test_yaml_loader.h"

#define DEFINE_FUNCTION_CLASS(FUNCTION)     \
class FUNCTION : public testing::TestWithParam<Arguments>    \
{                                                            \
protected:                                                   \
    FUNCTION() {}                                            \
    virtual ~FUNCTION() {}                                   \
    virtual void SetUp() {}                                  \
    virtual void TearDown() {}                               \
};

#define DEFINE_TEST_P(FUNCTION)                               \
TEST_P(FUNCTION, linear_solvers)                              \
{                                                             \
    EXPECT_TRUE(test_dispatch(GetParam()));                   \
}

#define DEFINE_INSTANTIATE_TEST_SUITE_P(FUNCTION, CATRGORY, YAML_TEST_FILE)  \
INSTANTIATE_TEST_SUITE_P(                                                    \
    quick,                                                                   \
    FUNCTION,                                                                \
    testing::ValuesIn(generate_tests(YAML_TEST_FILE)),                       \
    [](const testing::TestParamInfo<FUNCTION::ParamType>& info) {            \
        return info.param.generate_test_name();                              \
    });

#define INSTANTIATE_TEST(FUNCTION, CATEGORY, YAML_TEST_FILE)             \
namespace Testing                                                        \
{                                                                        \
    DEFINE_FUNCTION_CLASS(FUNCTION);                                     \
    DEFINE_TEST_P(FUNCTION);                                             \
    DEFINE_INSTANTIATE_TEST_SUITE_P(FUNCTION, CATRGORY, YAML_TEST_FILE); \
}

#endif
