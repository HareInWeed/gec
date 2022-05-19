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
using LIMB2_T = uint64_t;

/// @brief number of limbs for 160 bits bigint with 32-bit limb
constexpr size_t LN_160 = 5;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_160 = 0x96c9e927u;
/// @brief the modulus with 160 bits
alignas(8) extern const LIMB_T MOD_160[LN_160];
/// @brief R^2 (mod MOD)
alignas(8) extern const LIMB_T RR_160[LN_160];
/// @brief R (mod MOD)
alignas(8) extern const LIMB_T OneR_160[LN_160];

/// @brief number of limbs for 160 bits bigint with 64-bit limb
constexpr size_t LN2_160 = 3;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_160 = 0x1c23727c96c9e927u;
/// @brief the modulus with 160 bits
alignas(8) extern const LIMB2_T MOD2_160[LN2_160];
/// @brief R^2 (mod MOD)
alignas(8) extern const LIMB2_T RR2_160[LN2_160];
/// @brief R (mod MOD)
alignas(8) extern const LIMB2_T OneR2_160[LN2_160];

/// @brief number of limbs for 256 bits bigint with 32-bit limb
constexpr size_t LN_256 = 8;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_256 = 0xd2253531u;
/// @brief the modulus with 256 bits
alignas(32) extern const LIMB_T MOD_256[LN_256];
/// @brief R^2 (mod MOD)
alignas(32) extern const LIMB_T RR_256[LN_256];
/// @brief R (mod MOD)
alignas(32) extern const LIMB_T OneR_256[LN_256];

/// @brief number of limbs for 256 bits bigint with 64-bit limb
constexpr size_t LN2_256 = 4;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_256 = 0xd838091dd2253531u;
/// @brief the modulus with 256 bits
alignas(32) extern const LIMB2_T MOD2_256[LN2_256];
/// @brief R^2 (mod MOD)
alignas(32) extern const LIMB2_T RR2_256[LN2_256];
/// @brief R (mod MOD)
alignas(32) extern const LIMB2_T OneR2_256[LN2_256];

#endif // !GEC_TEST_COMMON_HPP
