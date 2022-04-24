#pragma once
#ifndef GEC_TEST_COMMON_HPP
#define GEC_TEST_COMMON_HPP

#include <cinttypes>
#include <iostream>

// #define GEC_DEBUG
#define GEC_IOSTREAM

using namespace std;

using LIMB_T = uint32_t;
constexpr size_t LIMB_N = 5;
extern const LIMB_T MOD[LIMB_N];

#endif // !GEC_TEST_COMMON_HPP