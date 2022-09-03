#pragma once
#ifndef GEC_TEST_COMMON_HPP
#define GEC_TEST_COMMON_HPP

// #define GEC_DEBUG

#include <gec/bigint.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/macros.hpp>

#include <iostream>

using namespace std;

using LIMB_T = uint32_t;
using LIMB2_T = uint64_t;

/// @brief number of limbs for 160 bits bigint with 32-bit limb
constexpr size_t LN_160 = 5;
/// @brief array for 160 bits bigint with 32-bit limb
using Array160 = gec::bigint::ArrayBE<LIMB_T, LN_160, 8>;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_160 = 0x96c9e927u;
/// @brief the modulus with 160 bits
GEC_DECL_GLOBAL(MOD_160, Array160);
/// @brief R^2 (mod MOD)
GEC_DECL_GLOBAL(RR_160, Array160);
/// @brief R (mod MOD)
GEC_DECL_GLOBAL(OneR_160, Array160);

/// @brief number of limbs for 160 bits bigint with 64-bit limb
constexpr size_t LN2_160 = 3;
/// @brief array for 160 bits bigint with 64-bit limb
using Array160_2 = gec::bigint::ArrayBE<LIMB2_T, LN2_160, 8>;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_160 = 0x1c23727c96c9e927llu;
/// @brief the modulus with 160 bits
GEC_DECL_GLOBAL(MOD2_160, Array160_2);
/// @brief R^2 (mod MOD)
GEC_DECL_GLOBAL(RR2_160, Array160_2);
/// @brief R (mod MOD)
GEC_DECL_GLOBAL(OneR2_160, Array160_2);

/// @brief number of limbs for 256 bits bigint with 32-bit limb
constexpr size_t LN_256 = 8;
/// @brief array for 256 bits bigint with 32-bit limb
using Array256 = gec::bigint::ArrayBE<LIMB_T, LN_256, 32>;
/// @brief -MOD^{-1} (mod 2^32)
constexpr LIMB_T MOD_P_256 = 0xd2253531u;
/// @brief the modulus with 256 bits
GEC_DECL_GLOBAL(MOD_256, Array256);
/// @brief R^2 (mod MOD)
GEC_DECL_GLOBAL(RR_256, Array256);
/// @brief R (mod MOD)
GEC_DECL_GLOBAL(OneR_256, Array256);

/// @brief number of limbs for 256 bits bigint with 64-bit limb
constexpr size_t LN2_256 = 4;
/// @brief array for 256 bits bigint with 64-bit limb
using Array256_2 = gec::bigint::ArrayBE<LIMB2_T, LN2_256, 8>;
/// @brief -MOD^{-1} (mod 2^64)
constexpr LIMB2_T MOD2_P_256 = 0xd838091dd2253531llu;
/// @brief the modulus with 256 bits
GEC_DECL_GLOBAL(MOD2_256, Array256_2);
/// @brief R^2 (mod MOD)
GEC_DECL_GLOBAL(RR2_256, Array256_2);
/// @brief R (mod MOD)
GEC_DECL_GLOBAL(OneR2_256, Array256_2);

#endif // !GEC_TEST_COMMON_HPP
