#include "matrix_multiplication.h"
#include "../src/matrix_mult.cpp"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult

// In this file we're going to briefly describe the behaviour of the function multiplyMatrices.
// In particular, for each test we're going to list the errors that we discovered (below the test code).

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
// From this given example we get four errors:
// * Error 6: Result matrix contains a number bigger than 100!
// * Error 12: The number of rows in A is equal to the number of columns in B!
// * Error 14: The result matrix C has an even number of rows!
// * Error 20: Number of columns in matrix A is odd!

// We can infer that the test will fail any time:
// * The matrix C contains at least one element greater than 100.
// * The matrix C is a square matrix because rowsC = rowsA = colsB = colsC.
// * The matrix C has an even number of rows.
// * The matrix A has an odd number of columns, therefore the matrix B has an odd number of rows for compatibility of dimensions.

// In order to detect all the other errors, we can start with the simplest possible case: multiplication between two scalars.
// We expect errors 12 and 20.
TEST(MatrixMultiplicationTest1, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1}
    };
    std::vector<std::vector<int>> B = {
        {2}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 1, 1, 1);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 12.
// * Error 18: Matrix A is a square matrix!
// * Error 20.

// We can infer that the test will fail any time the matrix A is a square matrix.

// So we can check again error 12 by multiplying a 2x2 matric by a 2x1 vector. We expect errors 14 and 18.
TEST(MatrixMultiplicationTest2, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2},
        {1, 2}
    };
    std::vector<std::vector<int>> B = {
        {3},
        {3}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 2, 2, 1);

    std::vector<std::vector<int>> expected(2, std::vector<int>(1, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 2, 2, 1);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 4: Matrix B contains the number 3!
// * Error 14.
// * Error 18.

// We can infer that the test will fail any time the matrix B contains a certain number, the same could happen for matrix A,
// so we can try to fill both A and B with all decimal places in order to detect other errors of the same type.
TEST(MatrixMultiplicationTest3, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 0, 1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(4, 0));

    multiplyMatrices(A, B, C, 2, 3, 4);

    std::vector<std::vector<int>> expected(2, std::vector<int>(4, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 2, 3, 4);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1: Element-wise multiplication of ones detected!
// * Error 4.
// * Error 7: Result matrix contains a number between 11 and 20!
// * Error 14.
// * Error 16: Matrix B contains the number 6!
// * Error 20.

// We can infer that the test will fail any time:
// * The current element in A and the current element in B are both one, so you get an element-wise multiplication.
// * The matrix C contains at least one element between 11 and 20.
// * The matrix B contains 3 and 6.

// Let's see what happens if we switch the previous A and B. We expect errors 1, 4, 7, 16, 20.
TEST(MatrixMultiplicationTest4, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 4, 8},
        {1, 5, 9},
        {2, 6, 0},
        {3, 7, 1}
    };
    std::vector<std::vector<int>> B = {
        {1, 4},
        {2, 5},
        {3, 6}
    };
    std::vector<std::vector<int>> C(4, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 4, 3, 2);

    std::vector<std::vector<int>> expected(4, std::vector<int>(2, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 4, 3, 2);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1.
// * Error 2: Matrix A contains the number 7!
// * Error 4.
// * Error 7.
// * Error 14.
// * Error 16.
// * Error 20.

// What if we multiply by the identity matrix?
// What if we set a wrong number of columns of B?
TEST(MatrixMultiplicationTest5, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> B = {
        {1, 1},
        {1, 1},
        {1, 1}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    std::vector<std::vector<int>> expected(3, std::vector<int>(3, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 3, 3, 3);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1.
// * Error 8: Result matrix contains zero!
// * Error 13: The first element of matrix A is equal to the first element of matrix B!
// * Error 18.
// * Error 20.

// ...

// What if we multiply by the null matrix?
TEST(MatrixMultiplicationTest6, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {0, 0},
        {0, 0},
        {0, 0}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected(2, std::vector<int>(2, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 2, 3, 2);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 8.
// * Error 11: Every row in matrix B contains at least one '0'!
// * Error 12.
// * Error 14.
// * Error 20.

// ...

// What if we define some negative elements?
TEST(MatrixMultiplicationTest7, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {-1, -1, -1},
        {-2, -2, -2}
    };
    std::vector<std::vector<int>> B = {
        {-1, 2, 3},
        {-4, 5, 6},
        {-7, 8, 9}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 2, 3, 3);

    std::vector<std::vector<int>> expected(2, std::vector<int>(3, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 2, 3, 3);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 3: Matrix A contains a negative number!
// * Error 4.
// * Error 5: Matrix B contains a negative number!
// * Error 8.
// * Error 13.
// * Error 14.
// * Error 16.
// * Error 20.

// ...

TEST(MatrixMultiplicationTest8, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 0, 0},
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {6, 6, 6},
        {7, 7, 7},
        {8, 8, 8},
        {9, 9, 9}
    };
    std::vector<std::vector<int>> B = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    std::vector<std::vector<int>> C(10, std::vector<int>(10, 0));

    multiplyMatrices(A, B, C, 10, 3, 10);

    std::vector<std::vector<int>> expected(10, std::vector<int>(10, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 10, 3, 10);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1.
// * Error 2.
// * Error 4.
// * Error 6.
// * Error 7.
// * Error 8.
// * Error 10.
// * Error 11.
// * Error 12.
// * Error 13.
// * Error 14.
// * Error 15: A row in matrix A is filled entirely with 5s!
// * Error 16.
// * Error 17: Result matrix C contains the number 17!
// * Error 20.

// ...

TEST(MatrixMultiplicationTest9, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    std::vector<std::vector<int>> B = {
        {0, 0, 0},
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {6, 6, 6},
        {7, 7, 7},
        {8, 8, 8},
        {9, 9, 9}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 10, 3);

    std::vector<std::vector<int>> expected(3, std::vector<int>(3, 0));

    multiplyMatricesWithoutErrors(A, B, expected, 3, 10, 3);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
// * Error 1.
// * Error 2.
// * Error 4.
// * Error 6.
// * Error 12.
// * Error 13.
// * Error 16.
// * Error 19: Every row in matrix A contains the number 8!

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
