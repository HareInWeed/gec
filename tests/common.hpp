#pragma once
#ifndef GEC_TEST_COMMON_HPP
#define GEC_TEST_COMMON_HPP

#include <cinttypes>
#include <iostream>
#include <limits>

#define GEC_DEBUG
#define GEC_IOSTREAM

using namespace std;

using LIMB_T = uint32_t;

/// @brief number of limbs for 160 bits bigint
constexpr size_t LN_160 = 5;
/// @brief the modulus with 160 bits
extern const LIMB_T MOD_160[LN_160];
/// @brief -MOD^{-1} (mod limb_bits)
constexpr LIMB_T MOD_P160 = 0x96c9e927u;
/// @brief R^2 (mod MOD)
extern const LIMB_T RR_160[LN_160];

/// @brief number of limbs for 256 bits bigint
constexpr size_t LN_256 = 8;
/// @brief the modulus with 256 bits
alignas(32) extern const LIMB_T MOD_256[LN_256];
/// @brief -MOD^{-1} (mod limb_bits)
constexpr LIMB_T MOD_P_256 = 0xd2253531u;
/// @brief R^2 (mod MOD)
alignas(32) extern const LIMB_T RR_256[LN_256];

#endif // !GEC_TEST_COMMON_HPP
