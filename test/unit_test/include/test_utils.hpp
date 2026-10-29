#pragma once

// gtest-lite assertion helpers for image comparisons. Shared non-test
// utilities (FillDeterministic, CheckCorrectness, RandomInt) live in the
// library's riscv_utils.hpp.

#include "gtest_lite.hpp"
#include "riscv_utils.hpp"

template <typename T>
void ExpectImagesEqual(const Image<T>& expected, const Image<T>& actual)
{
    EXPECT_TRUE(CheckCorrectness(expected, actual));
}