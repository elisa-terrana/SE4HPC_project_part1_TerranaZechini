#include "matrix_multiplication.h"
#include "../src/matrix_mult.cpp"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult


TEST(MatrixMultiplicationTest, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// in aggiunta ai commenti sul branch elisa

// come test 7...
// --------------------------
// Then we follow up with matrices containing negative numbers to see if that causes any issues
TEST(MatrixMultiplicationTest2, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, -1},
        {-1, 1}
    };
    std::vector<std::vector<int>> B = {
        {-1, -1},
        {-1, -1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 2, 2);

    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));
    multiplyMatricesWithoutErrors(A, B, expected, 2, 2, 2);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// --------------------------

// metterei questo tra test 2 e test 3, da qui ----------
// By trying again with small martices containing lots of zeros and a three in B, we expect errors 4, 12, 14 and 18
TEST(MatrixMultiplicationTest2_1, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 0},
        {5, 5}
    };
    std::vector<std::vector<int>> B = {
        {0, 3},
        {0, 3}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 2, 2);

    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));
    multiplyMatricesWithoutErrors(A, B, expected, 2, 2, 2);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 4.
// * Error 7: Result matrix contains a number between 11 and 20!
// * Error 8: Result matrix contains zero!
// * Error 11: Every row in matrix B contains at least one '0'!
// * Error 12.
// * Error 13: The first element of matrix A is equal to the first element of matrix B!
// * Error 14.
// * Error 15: A row in matrix A is filled entirely with 5s!
// * Error 18.

// From here we understand that we neen to explore the behaviour of the function by changing:
// - the size of the matrices
// - the magnitude of the values of the matrices
// - the presence of various digits within the matrices
// a qui --------------------------------------------

// come test 5...
// --------------------------
// in test 3 we multiply by the identity matrix to see if the result doesn't change
TEST(MatrixMultiplicationTest3, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 1},
        {1, 7}
    };
    std::vector<std::vector<int>> B = {
        {1, 0},
        {0, 1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 2, 2);

    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));
    multiplyMatricesWithoutErrors(A, B, expected, 2, 2, 2);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// --------------------------

// sostituirei test 4 con questo
// test truncation-matrix, the expected errors are 1, 7, 11, 12, 13, 14.
TEST(MatrixMultiplicationTest4, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 3, 5, 7, 9, 11},
        {2, 6, 10, 14, 18, 22},
        {98, 94, 90, 86, 82, 78},
        {99, 97, 95, 93, 91, 89}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    std::vector<std::vector<int>> C(4, std::vector<int>(4, 0));

    multiplyMatrices(A, B, C, 4, 6, 4);

    std::vector<std::vector<int>> expected(4, std::vector<int>(4, 0));
    multiplyMatricesWithoutErrors(A, B, expected, 4, 6, 4);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1.
// * Error 2: Matrix A contains the number 7!
// * Error 7.
// * Error 9: Result matrix contains the number 99!
// * Error 11.
// * Error 12.
// * Error 13.
// * Error 14.
// With this test we found two more cases in witch a specific number present in an input or output matrix causes an error.

// Finally, we multiply bigger matrices to see if more big valus cause any more errors
TEST(MatrixMultiplicationTest10, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0},
        {1, 0, 1, 0, 1},
    };
    std::vector<std::vector<int>> B = {
        {10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
        {110, 120, 130, 140, 150, 160, 170, 180, 190, 200},
        {210, 220, 230, 240, 250, 260, 270, 280, 290, 300},
        {310, 320, 330, 340, 350, 360, 370, 380, 390, 400},
        {410, 420, 430, 440, 450, 460, 470, 480, 490, 500}
    };
    std::vector<std::vector<int>> C(4, std::vector<int>(10, 0));

    multiplyMatrices(A, B, C, 4, 5, 10);

    std::vector<std::vector<int>> expected(4, std::vector<int>(10, 0));
    multiplyMatricesWithoutErrors(A, B, expected, 4, 5, 10);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// The test doesn't show any additional errors so we can move on to the next step.


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
