#pragma once
#ifndef GEC_TEST_COMMON_HPP
#define GEC_TEST_COMMON_HPP

#include <cinttypes>
#include <iostream>

#define GEC_DEBUG
#define GEC_IOSTREAM

using namespace std;

using LIMB_T = uint32_t;

/// @brief number of limbs for 160bit bigint
constexpr size_t LN_160 = 5;

extern const LIMB_T MOD160[LN_160];

/// @brief MOD_P: -MOD^{-1} (mod limb_bits)
constexpr LIMB_T MOD160_P = 0x96c9e927u;

/// @brief R2: R^2 (mod MOD)
extern const LIMB_T R160_SQR[LN_160];

#endif // !GEC_TEST_COMMON_HPP