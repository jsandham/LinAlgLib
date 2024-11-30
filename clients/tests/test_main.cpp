#include <iostream>

#include <gtest/gtest.h>

#include "test_iterative.h"
#include "test_pcg.h"
#include "test_saamg.h"
#include "test_rsamg.h"


std::string  jacobi_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class jacobi_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    jacobi_parameters() {}
    virtual ~jacobi_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(jacobi_parameters, jacobi_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_jacobi(filename));
}

INSTANTIATE_TEST_CASE_P(jacobi,
                        jacobi_parameters,
                        testing::Combine(testing::ValuesIn(jacobi_matrix_files)));

std::string  gauss_seidel_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class gauss_seidel_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    gauss_seidel_parameters() {}
    virtual ~gauss_seidel_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(gauss_seidel_parameters, gauss_seidel_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_gauss_seidel(filename));
}

INSTANTIATE_TEST_CASE_P(gauss_seidel,
                        gauss_seidel_parameters,
                        testing::Combine(testing::ValuesIn(gauss_seidel_matrix_files)));


std::string  sor_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class sor_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    sor_parameters() {}
    virtual ~sor_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(sor_parameters, sor_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_sor(filename));
}

INSTANTIATE_TEST_CASE_P(sor,
                        sor_parameters,
                        testing::Combine(testing::ValuesIn(sor_matrix_files)));

std::string  sym_gauss_seidel_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class sym_gauss_seidel_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    sym_gauss_seidel_parameters() {}
    virtual ~sym_gauss_seidel_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(sym_gauss_seidel_parameters, sym_gauss_seidel_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_symmetric_gauss_seidel(filename));
}

INSTANTIATE_TEST_CASE_P(sym_gauss_seidel,
                        sym_gauss_seidel_parameters,
                        testing::Combine(testing::ValuesIn(sym_gauss_seidel_matrix_files)));

std::string  ssor_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class ssor_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    ssor_parameters() {}
    virtual ~ssor_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(ssor_parameters, ssor_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_ssor(filename));
}

INSTANTIATE_TEST_CASE_P(ssor,
                        ssor_parameters,
                        testing::Combine(testing::ValuesIn(ssor_matrix_files)));

std::string  pcg_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                "../clients/matrices/bcsstm05.mtx",
                                "../clients/matrices/bcsstm22.mtx",
                                "../clients/matrices/bodyy4.mtx",
                                "../clients/matrices/bodyy5.mtx",
                                "../clients/matrices/bodyy6.mtx",
                                "../clients/matrices/nos1.mtx",
                                "../clients/matrices/nos6.mtx",
                                "../clients/matrices/ex5.mtx",
                                "../clients/matrices/crystm02.mtx",
                                "../clients/matrices/fv1.mtx",
                                "../clients/matrices/fv2.mtx",
                                "../clients/matrices/fv3.mtx",
                                "../clients/matrices/mesh1em6.mtx",
                                "../clients/matrices/mesh2em5.mtx",
                                "../clients/matrices/mesh3em5.mtx",
                                "../clients/matrices/ted_B.mtx",
                                "../clients/matrices/test.mtx",
                                "../clients/matrices/shallow_water1.mtx",
                                "../clients/matrices/shallow_water2.mtx"};

class pcg_parameters : public testing::TestWithParam<std::tuple<std::string>>
{
protected:
    pcg_parameters() {}
    virtual ~pcg_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(pcg_parameters, pcg_test)
{
    std::string filename = std::get<0>(GetParam());
    EXPECT_EQ(1, Testing::test_pcg(filename));
}

INSTANTIATE_TEST_CASE_P(pcg,
                        pcg_parameters,
                        testing::Combine(testing::ValuesIn(pcg_matrix_files)));


std::string  saamg_matrix_files[] = {"../clients/matrices/bcsstm02.mtx",
                                    "../clients/matrices/bcsstm05.mtx",
                                    "../clients/matrices/bcsstm22.mtx",
                                    "../clients/matrices/bodyy4.mtx",
                                    "../clients/matrices/bodyy5.mtx",
                                    "../clients/matrices/crystm02.mtx",
                                    "../clients/matrices/fv1.mtx",
                                    "../clients/matrices/fv2.mtx",
                                    "../clients/matrices/fv3.mtx",
                                    "../clients/matrices/mesh1em6.mtx",
                                    "../clients/matrices/mesh2em5.mtx",
                                    "../clients/matrices/mesh3em5.mtx",
                                    "../clients/matrices/ted_B.mtx",
                                    "../clients/matrices/test.mtx",
                                    "../clients/matrices/shallow_water1.mtx",
                                    "../clients/matrices/shallow_water2.mtx"};

int saamg_presmoothing_iterations[] = {2};
int saamg_postsmoothing_iterations[] = {4};
Cycle cycle_types[] = {Cycle::Fcycle, Cycle::Vcycle, Cycle::Wcycle};
Smoother smoother_type[] = {Smoother::Jacobi, Smoother::Gauss_Siedel, Smoother::SOR};

class saamg_parameters : public testing::TestWithParam<std::tuple<std::string, int, int, Cycle, Smoother>>
{
protected:
    saamg_parameters() {}
    virtual ~saamg_parameters() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(saamg_parameters, saamg_test)
{
    std::string filename = std::get<0>(GetParam());
    int presmoothing = std::get<1>(GetParam());
    int postsmoothing = std::get<2>(GetParam());
    Cycle cycle = std::get<3>(GetParam());
    Smoother smoother = std::get<4>(GetParam());
    EXPECT_EQ(1, Testing::test_saamg(filename, presmoothing, postsmoothing, cycle, smoother));
}

INSTANTIATE_TEST_CASE_P(saamg,
                        saamg_parameters,
                        testing::Combine(testing::ValuesIn(saamg_matrix_files),
                                         testing::ValuesIn(saamg_presmoothing_iterations),
                                         testing::ValuesIn(saamg_postsmoothing_iterations),
                                         testing::ValuesIn(cycle_types),
                                         testing::ValuesIn(smoother_type)));


int main()
{
    testing::InitGoogleTest();

    RUN_ALL_TESTS();

    return 0;
}