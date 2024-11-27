#include <iostream>

#include <gtest/gtest.h>

#include "test_pcg.h"
#include "test_saamg.h"
#include "test_rsamg.h"

// PCG tests
TEST(quick, pcg) {
    //Passing
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bcsstm02.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bcsstm05.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bcsstm22.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bodyy4.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bodyy5.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/bodyy6.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/nos1.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/nos6.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/ex5.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/crystm02.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/fv1.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/fv1.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/fv3.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/mesh1em6.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/mesh2em5.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/mesh3em5.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/ted_B.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/test.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/shallow_water1.mtx"));
    EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/shallow_water2.mtx"));

    // Slow convergence
    //EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/nos2.mtx"));
    //EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/cfd2.mtx"));
    //EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/Pres_Poisson.mtx"));
    //EXPECT_EQ(1, Testing::test_pcg("../clients/matrices/thermal1.mtx"));
}

// SAAMG tests
TEST(quick, saamg) {
    //Passing
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bcsstm02.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bcsstm05.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bcsstm22.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bodyy4.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bodyy5.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/crystm02.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/fv1.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/fv1.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/fv3.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/mesh1em6.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/mesh2em5.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/mesh3em5.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/ted_B.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/test.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/shallow_water1.mtx"));
    EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/shallow_water2.mtx"));

    // Slow convergence
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/bodyy6.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/cfd2.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/nos1.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/nos2.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/nos6.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/Pres_Poisson.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/ex5.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/thermal1.mtx"));

    // Failing convergence 
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/test2.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/ash85.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/thermomech_dK.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/thermomech_dM.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/thermomech_TC.mtx"));
    //EXPECT_EQ(1, Testing::test_saamg("../clients/matrices/thermomech_TK.mtx"));
}

// RSAMG tests
TEST(quick, rsamg) {
    
}

int main()
{
    testing::InitGoogleTest();

    RUN_ALL_TESTS();

    return 0;
}